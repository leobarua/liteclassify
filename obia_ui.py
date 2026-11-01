import numpy as np
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QGraphicsEllipseItem, QListWidget, QStyledItemDelegate, QStyle,
    QAbstractButton, QWidget, QVBoxLayout, QToolButton, QFrame, QHBoxLayout, QLabel, QDialog,
    QGridLayout, QCheckBox, QPushButton, QComboBox, QScrollArea, QListWidgetItem, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter, QMouseEvent, QKeyEvent, QPen, QBrush, QFont, QFontMetrics, QPolygonF, QPainterPath, QIcon, QTransform
from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QRect, QRectF, QSize, QTimer

# --- Custom List Item Highlight Delegate ---
class ClassListDelegate(QStyledItemDelegate):
    """
    Prevents Qt from drawing a dark selection highlight background over user class colors.
    Active classes display a clean 3px high-contrast cyan outline (matching ArcGIS-style
    selection highlighting), keeping their assigned colors visible. The canvas's own
    on-canvas segment-selection highlight (translucent cyan, see
    update_classification_display in obia_app.py) now intentionally matches this cyan too,
    per explicit request -- if a user-defined class happens to also be cyan, its training
    samples and an active selection can look the same at a glance; that tradeoff was
    accepted deliberately in favor of a single consistent selection color everywhere.
    """
    def paint(self, painter, option, index):
        painter.save()

        # Get Item background and text colors directly
        bg_brush = index.data(Qt.BackgroundRole)
        text_str = index.data(Qt.DisplayRole)
        font = index.data(Qt.FontRole)

        # Draw Base Color Block
        if bg_brush:
            painter.fillRect(option.rect, bg_brush)

        # Draw Border Outline if selected
        if option.state & QStyle.State_Selected:
            pen = QPen(QColor(0, 255, 255), 3, Qt.SolidLine)
            painter.setPen(pen)
            # Offset rect slightly inside to display a crisp outline
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))

        # Render Item Text
        painter.setPen(QColor(0, 0, 0)) # High contrast black text
        if font:
            painter.setFont(font)
        painter.drawText(option.rect.adjusted(10, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, text_str)
        
        painter.restore()


# --- Overlay layers ---
def rgba_premultiplied(color, alpha):
    """ (B, G, R, A) bytes, premultiplied, for ArrayLayerItem arrays (Qt's native
    ARGB32_Premultiplied layout on little-endian machines). """
    c = QColor(color)
    a = int(alpha)
    return (c.blue() * a // 255, c.green() * a // 255, c.red() * a // 255, a)


class ArrayLayerItem(QGraphicsItem):
    """ A (h, w, 4) uint8 BGRA-premultiplied numpy array drawn stretched over the base image.
    The QImage wraps the array's memory (no copy), so writing into the array and calling
    update_region() repaints just that part -- a click that changes one segment does not
    rebuild a full-size image. """

    def __init__(self, arr, base_w, base_h):
        super().__init__()
        self.arr = np.ascontiguousarray(arr)
        h, w = self.arr.shape[:2]
        self.img = QImage(self.arr.data, w, h, 4 * w, QImage.Format_ARGB32_Premultiplied)
        self.base_w, self.base_h = base_w, base_h
        self.sx, self.sy = base_w / w, base_h / h
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def boundingRect(self):
        return QRectF(0, 0, self.base_w, self.base_h)

    def paint(self, painter, option, widget=None):
        r = option.exposedRect.intersected(self.boundingRect())
        if r.isEmpty():
            return
        src = QRectF(r.x() / self.sx, r.y() / self.sy, r.width() / self.sx, r.height() / self.sy)
        painter.drawImage(r, self.img, src)

    def update_region(self, r0, r1, c0, c1):
        self.update(QRectF(c0 * self.sx, r0 * self.sy, (c1 - c0) * self.sx, (r1 - r0) * self.sy))


class BlendPixmapItem(QGraphicsPixmapItem):
    """ A pixmap layer that can draw with a composition mode other than normal alpha
    (e.g. Multiply, which tints the ortho underneath while keeping its texture). """

    def __init__(self):
        super().__init__()
        self.mode = QPainter.CompositionMode_SourceOver
        self._key = None

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setCompositionMode(self.mode)
        super().paint(painter, option, widget)
        painter.restore()


# --- Custom Graphics View for Panning, Zooming, and Painting ---
class InteractivePhotoViewer(QGraphicsView):
    """
    Custom graphics canvas supporting click selections, zooming,
    and a continuous click-and-drag "Paint Brush" tool when holding the Ctrl key.
    """
    photoClicked = Signal(QPoint)
    paintDragSegment = Signal(QPoint)
    paintDragFinished = Signal()
    segmentSelected = Signal(QPoint)
    selectDragSegment = Signal(QPoint)
    selectDragFinished = Signal()
    undoRequested = Signal()

    # --- "points" interaction mode (Accuracy Assessment tab) ---
    # Plain click toggles batch-select (mirrors "select" mode's click-to-toggle
    # convention exactly, just targeting a point instead of a segment). Right-click
    # opens a context menu -- previously-unused gesture, freed up because plain click
    # now does select-toggle duty. Ctrl+drag moves a single point (pick up on press,
    # live-update on drag, commit on release) rather than the paint-brush's "add
    # everything the cursor passes over" semantics.
    pointToggled = Signal(QPoint)
    pointContextMenuRequested = Signal(QPoint, QPoint)  # (pixmap_pos, global_screen_pos)
    pointMoveStarted = Signal(QPoint)
    pointMoveDragged = Signal(QPoint)
    pointMoveFinished = Signal(QPoint)
    # Right-click (no drag) in "paint"/"select" modes -- (pixmap_pos, global_screen_pos)
    contextMenuRequested = Signal(QPoint, QPoint)
    # Cursor position in grid pixels while hovering (status line), and leaving the map.
    cursorMoved = Signal(QPointF)
    cursorLeft = Signal()
    # Zoom or pan changed (shared view across tabs; see view_state / set_view_state).
    viewChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._layers = {}          # named overlay items over the base image, see set_layer
        self._markers = []         # point markers, see set_markers
        self._base_key = None
        # Map coordinates are the working grid's pixels (grid size), which the base pixmap
        # may show at a lower resolution on very large images -- it is then stretched over
        # the grid, and a sharp patch of the visible area (detail provider) is drawn on top
        # of it when zoomed in past the base pixmap's resolution.
        self._grid = None
        self._detail_item = QGraphicsPixmapItem()
        self._detail_item.setZValue(0.5)
        self._detail_item.setVisible(False)
        self._scene.addItem(self._detail_item)
        self._detail_provider = None
        self._detail_key = None
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(120)
        self._detail_timer.timeout.connect(self._update_detail)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self._is_painting = False
        self._interaction_mode = "paint"
        self._drag_start_pos = QPoint()
        self._right_click_start_pos = QPoint()
        self.setMouseTracking(True)            # cursor coordinates while hovering
        self.viewport().setMouseTracking(True)

    def set_interaction_mode(self, mode):
        """ "paint" (default) preserves today's Ctrl+drag paint-brush + photoClicked
        behavior exactly. "select" disables the paint-brush and emits segmentSelected
        on plain clicks instead of photoClicked. "points" is the Accuracy Assessment
        tab's mode -- plain click emits pointToggled, right-click emits
        pointContextMenuRequested, Ctrl+drag emits pointMoveStarted/Dragged/Finished. """
        assert mode in ("paint", "select", "points")
        self._interaction_mode = mode
        self._is_painting = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_image(self, qimage, grid_size=None):
        """ New base image (QImage or a shared QPixmap), fitted to the view; clears layers.
        grid_size=(w, h): the map's size in working-grid pixels when the image is a
        reduced display copy (default: the image's own size). """
        pixmap = qimage if isinstance(qimage, QPixmap) else QPixmap.fromImage(qimage)
        self._pixmap_item.setPixmap(pixmap)
        self._apply_grid(grid_size)
        self._base_key = None
        self.clear_layers()
        self.set_markers([])
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def update_image(self, qimage, grid_size=None):
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item.setPixmap(pixmap)
        self._apply_grid(grid_size)
        self._base_key = None

    def set_base_image_cached(self, image, grid_size=None):
        """ Shows image (QImage or QPixmap) as the base layer. A QPixmap is shared, not
        copied (one pixmap can back several viewers); either way nothing is redone when it
        is the same image as last time. grid_size as in set_image. """
        key = image.cacheKey()
        if key != self._base_key:
            self._pixmap_item.setPixmap(image if isinstance(image, QPixmap) else QPixmap.fromImage(image))
            self._apply_grid(grid_size)
            self._base_key = key

    def _apply_grid(self, grid_size):
        pm = self._pixmap_item.pixmap()
        pw, ph = pm.width(), pm.height()
        self._grid = (int(grid_size[0]), int(grid_size[1])) if grid_size else (pw, ph)
        gw, gh = self._grid
        if pw > 0 and ph > 0:
            self._pixmap_item.setTransform(QTransform.fromScale(gw / pw, gh / ph))
        self._detail_key = None
        self._detail_item.setVisible(False)
        self._detail_item.setPixmap(QPixmap())
        self._detail_timer.start()

    def base_size(self):
        """ The map's size in working-grid pixels (the coordinates every signal uses). """
        if self._grid is None:
            pm = self._pixmap_item.pixmap()
            return pm.width(), pm.height()
        return self._grid

    def _grid_rect(self):
        if self._pixmap_item.pixmap().isNull():
            return QRectF().toRect()
        w, h = self.base_size()
        return QRectF(0, 0, w, h).toRect()

    def set_detail_provider(self, fn):
        """ fn(x0, y0, x1, y1, step) -> QImage of grid pixels [y0:y1:step, x0:x1:step], or
        None. Called (debounced) when the view is zoomed in past the base pixmap's own
        resolution; the result is drawn over the base image, under every layer. """
        self._detail_provider = fn
        self._detail_key = None
        self._detail_timer.start()

    def _update_detail(self):
        it = self._detail_item
        pm = self._pixmap_item.pixmap()
        if self._detail_provider is None or pm.isNull():
            it.setVisible(False)
            return
        gw, gh = self.base_size()
        density = pm.width() / gw if gw else 1.0       # base pixmap px per grid px
        zoom = self.transform().m11()                  # screen px per grid px
        if density >= 0.999 or zoom <= density * 1.25:
            it.setVisible(False)
            return
        r = self.mapToScene(self.viewport().rect()).boundingRect().intersected(QRectF(0, 0, gw, gh))
        if r.isEmpty():
            it.setVisible(False)
            return
        x0, y0 = max(0, int(r.left()) - 2), max(0, int(r.top()) - 2)
        x1, y1 = min(gw, int(r.right()) + 3), min(gh, int(r.bottom()) + 3)
        step = max(1, int(1.0 / zoom))                 # no finer than the screen needs
        key = (x0, y0, x1, y1, step, self._base_key)
        if key != self._detail_key:
            img = self._detail_provider(x0, y0, x1, y1, step)
            if img is None or img.isNull():
                it.setVisible(False)
                return
            it.setPixmap(QPixmap.fromImage(img))
            it.setTransform(QTransform.fromScale((x1 - x0) / img.width(), (y1 - y0) / img.height()))
            it.setPos(x0, y0)
            self._detail_key = key
        it.setVisible(True)

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self._detail_timer.start()
        self.viewChanged.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._detail_timer.start()

    # --- shared view ---------------------------------------------------------------
    def view_state(self):
        """ (centre x, centre y in grid pixels, scale = screen px per grid px, grid size),
        or None without an image. """
        if self._pixmap_item.pixmap().isNull() or self.viewport().width() <= 0:
            return None
        c = self.mapToScene(self.viewport().rect().center())
        return (c.x(), c.y(), self.transform().m11(), tuple(self.base_size()))

    def set_view_state(self, state):
        """ Shows the same centre and scale as another viewer of the same grid. """
        if not state or self._pixmap_item.pixmap().isNull() or tuple(self.base_size()) != tuple(state[3]):
            return False
        x, y, s, _ = state
        fit = self._fit_scale()
        if fit is not None:
            s = max(s, fit)
        self.blockSignals(True)
        try:
            self.resetTransform()
            self.scale(s, s)
            self.centerOn(QPointF(x, y))
        finally:
            self.blockSignals(False)
        self._detail_timer.start()
        return True

    def leaveEvent(self, event):
        self.cursorLeft.emit()
        super().leaveEvent(event)

    def _place(self, item, w, h, z, opacity, rect=None):
        bw, bh = self.base_size()
        if rect is None:
            item.setTransform(QTransform.fromScale(bw / w, bh / h))
            item.setPos(0, 0)
        else:
            x, y, rw, rh = rect
            item.setTransform(QTransform.fromScale(rw / w, rh / h))
            item.setPos(x, y)
        item.setZValue(z)
        item.setOpacity(opacity)
        item.setVisible(opacity > 0)

    def set_layer(self, name, image, opacity=1.0, z=1, rect=None, multiply=False):
        """ A named overlay (QImage) stretched over the base image, or over rect=(x, y, w, h)
        in image pixels. Its pixmap is only rebuilt when the image changes, so calling this
        again with the same image just updates opacity/visibility. image=None removes it. """
        it = self._layers.get(name)
        if image is None or image.isNull() or self._pixmap_item.pixmap().isNull():
            if it is not None:
                self._scene.removeItem(it)
                del self._layers[name]
            return
        if it is not None and not isinstance(it, BlendPixmapItem):
            self._scene.removeItem(it)
            it = None
        if it is None:
            it = BlendPixmapItem()
            it.setTransformationMode(Qt.SmoothTransformation)
            self._scene.addItem(it)
            self._layers[name] = it
        if it._key != image.cacheKey():
            it.setPixmap(QPixmap.fromImage(image))
            it._key = image.cacheKey()
        mode = QPainter.CompositionMode_Multiply if multiply else QPainter.CompositionMode_SourceOver
        if it.mode != mode:
            it.mode = mode
            it.update()
        self._place(it, image.width(), image.height(), z, opacity, rect)

    def set_array_layer(self, name, arr, opacity=1.0, z=1):
        """ Replaces a named layer with an ArrayLayerItem over arr (see ArrayLayerItem). """
        self.set_layer(name, None)
        if arr is None or self._pixmap_item.pixmap().isNull():
            return None
        bw, bh = self.base_size()
        it = ArrayLayerItem(arr, bw, bh)
        it.setZValue(z)
        it.setOpacity(opacity)
        it.setVisible(opacity > 0)
        self._scene.addItem(it)
        self._layers[name] = it
        return it

    def layer(self, name):
        return self._layers.get(name)

    def set_layer_opacity(self, name, opacity):
        it = self._layers.get(name)
        if it is not None:
            it.setOpacity(opacity)
            it.setVisible(opacity > 0)

    def clear_layers(self):
        for it in self._layers.values():
            self._scene.removeItem(it)
        self._layers = {}

    def set_overlays(self, layers):
        """ layers: list of (QImage, opacity), stretched over the base image (Tab 1). """
        for i, (img, opacity) in enumerate(layers):
            self.set_layer(f"overlay{i}", img, opacity, z=1 + i)
        for i in range(len(layers), 4):
            self.set_layer(f"overlay{i}", None)

    def set_markers(self, markers, opacity=0.7):
        """ markers: list of (x, y, radius, fill QColor, pen QColor, ring bool); x, y in image
        pixels, radius in SCREEN pixels -- the markers keep the same on-screen size at any
        zoom (visible when zoomed out, never covering the land when zoomed in). """
        for m in self._markers:
            self._scene.removeItem(m)
        self._markers = []
        for x, y, r, fill, pen, ring in markers:
            e = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            e.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            e.setPos(x, y)
            e.setBrush(QBrush(fill))
            e.setPen(QPen(pen, 1.5))
            e.setOpacity(opacity)
            e.setZValue(100)
            self._scene.addItem(e)
            self._markers.append(e)
            if ring:
                rr = r + 3
                ring_item = QGraphicsEllipseItem(-rr, -rr, 2 * rr, 2 * rr)
                ring_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                ring_item.setPos(x, y)
                ring_item.setPen(QPen(QColor(0, 255, 255), 2))
                ring_item.setZValue(101)
                self._scene.addItem(ring_item)
                self._markers.append(ring_item)

    def fit_view(self):
        """ Explicit one-shot "zoom out to show the whole image" action, distinct from
        update_image (used for live overlay refreshes during interactive work -- must
        NOT reset the user's current zoom/pan) and set_image (used when swapping to a
        different image entirely, e.g. a fresh segmentation preview, which already
        fits by itself). Callers trigger this once right after a genuinely new import
        or resample, so the full image is visible on load instead of whatever zoom
        level a previous image happened to leave the view at. """
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._detail_timer.start()
            self.viewChanged.emit()

    def center_on_pixmap_point(self, x, y, zoom=None):
        """ Pans (and, if zoom is given, first sets an absolute zoom level) so the
        given point in pixmap/image-pixel space is centered in the viewport. Used by
        the Accuracy Assessment tab's `<`/`>` point navigation to jump to a comfortable
        inspection view of each point in turn. """
        if self._pixmap_item.pixmap().isNull():
            return
        scene_pos = QPointF(x, y)   # scene coordinates are grid pixels
        if zoom is not None:
            self.resetTransform()
            self.scale(zoom, zoom)
        self.centerOn(scene_pos)
        self._detail_timer.start()
        self.viewChanged.emit()

    def _fit_scale(self):
        """ The scale factor at which the loaded pixmap exactly fills the viewport
        (matching what fitInView computes) -- used as the zoom-out floor so the image
        can never be shrunk down to a speck. Recomputed live off the current viewport
        size rather than cached, so it stays correct across window resizes. """
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return None
        pw, ph = self.base_size()
        vw, vh = self.viewport().width(), self.viewport().height()
        if pw <= 0 or ph <= 0 or vw <= 0 or vh <= 0:
            return None
        return min(vw / pw, vh / ph)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        self._detail_timer.start()
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
            self.viewChanged.emit()
            return

        fit_scale = self._fit_scale()
        if fit_scale is None:
            self.scale(zoom_out_factor, zoom_out_factor)
            return

        current_scale = self.transform().m11()
        if current_scale <= fit_scale * 1.001:
            # Already at (or past) the "whole image fits the view" boundary -- ignore
            # further zoom-out instead of letting the image shrink into a speck.
            return
        if current_scale * zoom_out_factor < fit_scale:
            # This step would overshoot past the boundary -- snap exactly to it.
            self.resetTransform()
            self.scale(fit_scale, fit_scale)
            self.centerOn(self._pixmap_item)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)
        self.viewChanged.emit()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Ctrl+drag activates continuous painting in either mode -- which signal
            # it emits depends on the current interaction mode.
            if event.modifiers() & Qt.ControlModifier:
                self._is_painting = True
                self.setDragMode(QGraphicsView.NoDrag)
                scene_pos = self.mapToScene(event.pos())
                pixmap_pos = scene_pos.toPoint()
                if self._grid_rect().contains(pixmap_pos):
                    if self._interaction_mode == "select":
                        self.selectDragSegment.emit(pixmap_pos)
                    elif self._interaction_mode == "points":
                        self.pointMoveStarted.emit(pixmap_pos)
                    else:
                        self.paintDragSegment.emit(pixmap_pos)
            else:
                self._is_painting = False
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self._drag_start_pos = event.pos()
        elif event.button() == Qt.RightButton:
            # Not used by "paint"/"select" -- only "points" acts on this (see
            # mouseReleaseEvent). Recorded here either way so a subsequent release
            # always has a start position to measure drag distance against.
            self._right_click_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._pixmap_item.pixmap().isNull():
            self.cursorMoved.emit(self.mapToScene(event.pos()))
        if self._is_painting and (event.buttons() & Qt.LeftButton):
            scene_pos = self.mapToScene(event.pos())
            pixmap_pos = scene_pos.toPoint()
            if self._grid_rect().contains(pixmap_pos):
                if self._interaction_mode == "select":
                    self.selectDragSegment.emit(pixmap_pos)
                elif self._interaction_mode == "points":
                    self.pointMoveDragged.emit(pixmap_pos)
                else:
                    self.paintDragSegment.emit(pixmap_pos)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._is_painting:
            self._is_painting = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            scene_pos = self.mapToScene(event.pos())
            pixmap_pos = scene_pos.toPoint()
            if self._interaction_mode == "select":
                self.selectDragFinished.emit()
            elif self._interaction_mode == "points":
                self.pointMoveFinished.emit(pixmap_pos)
            else:
                self.paintDragFinished.emit()
        elif event.button() == Qt.LeftButton:
            # Safe distance threshold prevents selection triggers while panning
            drag_distance = (event.pos() - self._drag_start_pos).manhattanLength()
            if drag_distance < 15:
                scene_pos = self.mapToScene(event.pos())
                pixmap_pos = scene_pos.toPoint()
                if self._grid_rect().contains(pixmap_pos):
                    if self._interaction_mode == "select":
                        self.segmentSelected.emit(pixmap_pos)
                    elif self._interaction_mode == "points":
                        self.pointToggled.emit(pixmap_pos)
                    else:
                        self.photoClicked.emit(pixmap_pos)
        elif event.button() == Qt.RightButton:
            drag_distance = (event.pos() - self._right_click_start_pos).manhattanLength()
            if drag_distance < 15:
                scene_pos = self.mapToScene(event.pos())
                pixmap_pos = scene_pos.toPoint()
                if self._grid_rect().contains(pixmap_pos):
                    global_pos = self.viewport().mapToGlobal(event.pos())
                    if self._interaction_mode == "points":
                        self.pointContextMenuRequested.emit(pixmap_pos, global_pos)
                    else:
                        self.contextMenuRequested.emit(pixmap_pos, global_pos)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if self._interaction_mode == "select" and event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self.undoRequested.emit()
            return
        super().keyPressEvent(event)


# --- Pill-shaped checkable switch ---
class ElidedInfoLabel(QWidget):
    """ Right-aligned lines of small grey text that shorten with "…" when space runs out,
    and never ask for more width than they get (the map bar's image info and cursor line). """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines = []
        f = self.font()
        f.setPixelSize(11)
        self.setFont(f)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(1)

    def setText(self, text):
        lines = text.split("\n") if text else []
        if lines != self._lines:
            self._lines = lines
            self.updateGeometry()
            self.update()

    def text(self):
        return "\n".join(self._lines)

    def sizeHint(self):
        fm = self.fontMetrics()
        return QSize(max([fm.horizontalAdvance(l) for l in self._lines] or [0]), max(1, len(self._lines)) * fm.height())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setPen(QColor("#9aa59f"))
        fm = self.fontMetrics()
        h = fm.height()
        y0 = (self.height() - h * len(self._lines)) // 2
        for i, line in enumerate(self._lines):
            p.drawText(QRect(0, y0 + i * h, self.width(), h), Qt.AlignRight | Qt.AlignVCenter,
                       self._fit(line, fm))
        p.end()

    SEP = "  ·  "

    def _fit(self, line, fm):
        """ Drops whole items from the end ("a  ·  b  ·  c" -> "a  ·  b  …") rather than cutting
        a word; a single item that is still too long is elided. """
        w = self.width()
        if fm.horizontalAdvance(line) <= w:
            return line
        parts = line.split(self.SEP)
        while len(parts) > 1:
            parts.pop()
            cand = self.SEP.join(parts) + "  …"
            if fm.horizontalAdvance(cand) <= w:
                return cand
        return fm.elidedText(line, Qt.ElideRight, w)


class ToggleSwitch(QAbstractButton):
    """
    A checkable pill/switch control, used where a plain checkable QPushButton's
    text-only ON/OFF state was hard to read at a glance (e.g. "Show Training Samples").
    Checkable like any QAbstractButton -- isChecked()/setChecked()/toggled all work
    normally, this only replaces how the on/off state is drawn.
    """
    def __init__(self, parent=None, on_color=QColor(224, 92, 92), off_color=QColor(150, 154, 160)):
        super().__init__(parent)
        self.setCheckable(True)
        self._on_color = on_color
        self._off_color = off_color
        self.setCursor(Qt.PointingHandCursor)

    def sizeHint(self):
        return QSize(46, 24)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track_rect = QRectF(0, 0, self.width(), self.height())
        radius = track_rect.height() / 2.0

        track_color = self._on_color if self.isChecked() else self._off_color
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(track_rect, radius, radius)

        thumb_d = track_rect.height() - 4
        thumb_x = (track_rect.width() - thumb_d - 2) if self.isChecked() else 2
        thumb_rect = QRectF(thumb_x, 2, thumb_d, thumb_d)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(thumb_rect)

        painter.end()


# --- Collapsible wrapper for a bulky control group ---
class CollapsibleSection(QWidget):
    """
    Wraps an existing content widget (e.g. the Color Correction QGroupBox) behind a
    checkable QToolButton header that shows/hides it with a plain setVisible() -- no
    animation, since the content's height is dynamic (sliders/spinboxes) and animating
    that invites the usual QPropertyAnimation height-hunting bugs for no real benefit
    on a desktop app.
    """
    def __init__(self, title, content_widget, start_collapsed=True, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        self.toggle_btn.clicked.connect(self._on_toggled)

        self.content_widget = content_widget

        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.content_widget)

        self.toggle_btn.setChecked(not start_collapsed)
        self._on_toggled()

    def _on_toggled(self):
        expanded = self.toggle_btn.isChecked()
        self.content_widget.setVisible(expanded)
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

# --- Accuracy Assessment confusion-matrix figure ---
# Same look as the confusion matrix in the user manual (docs/manual/manual.css, ".cm"):
# paper background, hairline rules, green-tinted diagonal, tinted accuracy cells,
# muted totals, class swatches in the headers.
_CM_PAPER = QColor("#f8f9f6")
_CM_INK = QColor("#17201c")
_CM_MUTED = QColor("#76817b")
_CM_RULE = QColor("#dfe4de")
_CM_DIAG = QColor("#e3efe7")
_CM_TINT = QColor("#eef3ef")
_CM_UNCLASSIFIED = QColor("#bdbdbd")


def _cm_font(px, weight=QFont.Normal, display=False):
    font = QFont()
    font.setFamilies(["Archivo", "Segoe UI", "Arial"] if display else ["Public Sans", "Segoe UI", "Arial"])
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


def render_confusion_matrix_image(result, class_colors, scale=2):
    """ Draws the result of ObiaEngine.run_accuracy_assessment as a styled table:
    rows = map, columns = reference, a Total column/row, User's accuracy down the right,
    Producer's accuracy along the bottom, and overall accuracy in the corner. Returns a
    QImage at `scale`x the logical size so it stays sharp when dropped into a report.
    class_colors maps class name -> QColor (an "Unclassified" row gets neutral grey). """
    matrix = result['matrix']
    cols = list(matrix.columns)
    rows = list(matrix.index)
    ua, pa = result['users_accuracy'], result['producers_accuracy']
    oa = result['overall_accuracy']
    strat = result.get('stratified')
    ua_ci = pa_ci = {}
    oa_ci = None
    if strat is not None:
        # Stratified sample: the cells stay sample counts, the accuracies are area-weighted.
        ua, pa, oa = strat['ua'], strat['pa'], strat['oa']
        ua_ci = {c: (None if v is None else 1.96 * v) for c, v in strat['ua_se'].items()}
        pa_ci = {c: (None if v is None else 1.96 * v) for c, v in strat['pa_se'].items()}
        oa_ci = 1.96 * strat['oa_se']
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    grand_total = int(matrix.values.sum())

    def pct(v, decimals=0, ci=None):
        if v is None:
            return "—"
        return f"{v * 100:.{decimals}f}%" + ("" if ci is None else f" ±{ci * 100:.{decimals}f}")

    f_title = _cm_font(20, QFont.Bold, display=True)
    f_sub = _cm_font(13)
    f_head = _cm_font(14, QFont.DemiBold, display=True)
    f_corner = _cm_font(12, QFont.DemiBold, display=True)
    f_cell = _cm_font(15)
    f_bold = _cm_font(15, QFont.Bold)
    f_note = _cm_font(12)
    fm_head, fm_cell, fm_bold = QFontMetrics(f_head), QFontMetrics(f_cell), QFontMetrics(f_bold)

    margin, pad, row_h, head_h, sw = 28, 12, 40, 52, 12
    corner_lines = ["rows = map", "columns = reference"]
    first_w = max(
        [QFontMetrics(f_corner).horizontalAdvance(t) for t in corner_lines]
        + [sw + 8 + fm_head.horizontalAdvance(r) for r in rows]
        + [fm_head.horizontalAdvance("Producer’s")]
    ) + 2 * pad
    widest = "100% ±10" if strat is not None else "100%"
    col_ws = [max(sw + 8 + fm_head.horizontalAdvance(c), fm_bold.horizontalAdvance(widest)) + 2 * pad for c in cols]
    total_w = max(fm_head.horizontalAdvance("Total"), fm_cell.horizontalAdvance(str(grand_total))) + 2 * pad
    acc_w = max(fm_head.horizontalAdvance("User’s"),
                fm_bold.horizontalAdvance("OA 100.0% ±10.0" if strat is not None else "OA 100.0%")) + 2 * pad
    widths = [first_w] + col_ws + [total_w, acc_w]
    table_w = sum(widths)
    table_h = head_h + (len(rows) + 2) * row_h

    sources = result.get('source_counts') or {}
    note = f"{result['n_labeled']} labeled reference points"
    if sources:
        note += " (" + "; ".join(f"{n} {src}" for src, n in sources.items()) + ")"
    note += ". Unassigned points are excluded."
    if 'expert_visual' in sources:
        note += (" 'expert_visual' points come from independent visual interpretation of the "
                 "orthomosaic, not field verification.")
    if strat is not None:
        note += (" Stratified random sample (strata = map classes): cells are point counts; user's, "
                 "producer's and overall accuracy are area-weighted estimates with 95% confidence "
                 "intervals (Olofsson et al. 2014; Stehman 2014).")

    title_h = 58
    note_h = QFontMetrics(f_note).boundingRect(0, 0, table_w, 1000, Qt.TextWordWrap, note).height()
    img_w = table_w + 2 * margin
    img_h = margin + title_h + table_h + 14 + note_h + margin

    image = QImage(int(img_w * scale), int(img_h * scale), QImage.Format_ARGB32)
    image.fill(_CM_PAPER)
    p = QPainter(image)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.scale(scale, scale)

    # Title + subtitle
    p.setPen(_CM_INK)
    p.setFont(f_title)
    p.drawText(QRectF(margin, margin, table_w, 26), Qt.AlignLeft | Qt.AlignVCenter, "Accuracy Assessment")
    p.setPen(_CM_MUTED)
    p.setFont(f_sub)
    p.drawText(QRectF(margin, margin + 28, table_w, 20), Qt.AlignLeft | Qt.AlignVCenter,
               f"Confusion matrix · n = {grand_total} · overall accuracy {pct(oa, 1, oa_ci)}"
               + (" (area-weighted, 95% CI)" if strat is not None else ""))

    y0 = margin + title_h
    xs = [margin]
    for w in widths:
        xs.append(xs[-1] + w)

    def cell(i, j, text, font=f_cell, color=_CM_INK, fill=None, align=Qt.AlignRight, swatch=None):
        top = y0 if i == 0 else y0 + head_h + (i - 1) * row_h
        rect = QRectF(xs[j], top, xs[j + 1] - xs[j], head_h if i == 0 else row_h)
        if fill is not None:
            p.fillRect(rect, fill)
        p.setPen(QPen(_CM_RULE, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)
        inner = rect.adjusted(pad, 0, -pad, 0)
        if swatch is not None:
            sw_rect = QRectF(inner.left(), rect.center().y() - sw / 2, sw, sw)
            p.fillRect(sw_rect, swatch)
            p.setPen(QPen(swatch.darker(140), 1))
            p.drawRect(sw_rect)
            inner.setLeft(inner.left() + sw + 8)
        p.setPen(color)
        p.setFont(font)
        p.drawText(inner, align | Qt.AlignVCenter, text)

    def color_of(name):
        c = class_colors.get(name)
        return QColor(c) if c is not None else _CM_UNCLASSIFIED

    # Header row (the corner cell is borderless, as in the manual)
    p.setPen(_CM_MUTED)
    p.setFont(f_corner)
    p.drawText(QRectF(xs[0] + pad, y0, first_w - 2 * pad, head_h), Qt.AlignLeft | Qt.AlignVCenter,
               "\n".join(corner_lines))
    for j, c in enumerate(cols, start=1):
        cell(0, j, c, f_head, align=Qt.AlignLeft, swatch=color_of(c))
    cell(0, len(cols) + 1, "Total", f_head, _CM_MUTED)
    cell(0, len(cols) + 2, "User’s", f_head)

    # Body rows
    for i, r in enumerate(rows, start=1):
        cell(i, 0, r, f_head, align=Qt.AlignLeft, swatch=color_of(r))
        for j, c in enumerate(cols, start=1):
            is_diag = (r == c)
            cell(i, j, str(int(matrix.loc[r, c])), f_bold if is_diag else f_cell, fill=_CM_DIAG if is_diag else None)
        cell(i, len(cols) + 1, str(int(row_totals[r])), color=_CM_MUTED)
        # "Unclassified" (map-only bucket) has no User's accuracy
        if r in ua:
            cell(i, len(cols) + 2, pct(ua[r], ci=ua_ci.get(r)), f_bold, fill=_CM_TINT)
        else:
            cell(i, len(cols) + 2, "")

    # Totals row
    i_tot = len(rows) + 1
    cell(i_tot, 0, "Total", f_head, _CM_MUTED, align=Qt.AlignLeft)
    for j, c in enumerate(cols, start=1):
        cell(i_tot, j, str(int(col_totals[c])), color=_CM_MUTED)
    cell(i_tot, len(cols) + 1, str(grand_total), color=_CM_MUTED)
    cell(i_tot, len(cols) + 2, "")

    # Producer's row + overall accuracy
    i_pa = len(rows) + 2
    cell(i_pa, 0, "Producer’s", f_head, align=Qt.AlignLeft)
    for j, c in enumerate(cols, start=1):
        cell(i_pa, j, pct(pa[c], ci=pa_ci.get(c)), f_bold, fill=_CM_TINT)
    cell(i_pa, len(cols) + 1, "")
    cell(i_pa, len(cols) + 2, f"OA {pct(oa, 1, oa_ci)}", f_bold, fill=_CM_TINT)

    # Heavier rule under the header, as in the manual
    p.setPen(QPen(_CM_INK, 1.5))
    p.drawLine(QPointF(xs[1], y0 + head_h), QPointF(xs[-1], y0 + head_h))

    # Footnote
    p.setPen(_CM_MUTED)
    p.setFont(f_note)
    p.drawText(QRectF(margin, y0 + table_h + 14, table_w, note_h), Qt.AlignLeft | Qt.TextWordWrap, note)

    p.end()
    return image


# --- Class separability figure (same look as the confusion matrix) ---
_SEP_GOOD = QColor("#e3efe7")
_SEP_FAIR = QColor("#fbefd5")
_SEP_POOR = QColor("#f6dcd7")
JM_GOOD, JM_FAIR = 1.9, 1.7

FEATURE_LABELS = {
    'r_mean': "Red", 'r_std': "Red SD", 'g_mean': "Green", 'g_std': "Green SD", 'b_mean': "Blue",
    'b_std': "Blue SD", 'area': "Area", 'perimeter': "Perimeter", 'circularity': "Circularity",
    'convexity': "Convexity", 'contrast': "GLCM contrast", 'homogeneity': "GLCM homog.",
    'entropy': "GLCM entropy", 'energy': "GLCM energy", 'exg_mean': "ExG", 'vari_mean': "VARI",
    'ngrdi_mean': "NGRDI", 'brightness_mean': "Brightness", 'brightness_rel_neighbors': "Bright. vs nbrs",
    'dsm_mean': "Elevation", 'dsm_std': "Elev. SD", 'slope_mean': "Slope", 'dsm_rel_neighbors': "Elev. vs nbrs",
    'hag_mean': "Height above ground",
}


# Plain-language names, for the sentences in the separability figure.
PLAIN_LABELS = {
    'r_mean': "redness", 'r_std': "variation in red", 'g_mean': "greenness of colour", 'g_std': "variation in green",
    'b_mean': "blueness", 'b_std': "variation in blue", 'area': "size", 'perimeter': "outline length",
    'circularity': "roundness", 'convexity': "shape regularity", 'contrast': "texture contrast",
    'homogeneity': "texture smoothness", 'entropy': "texture complexity", 'energy': "texture uniformity",
    'exg_mean': "greenness (ExG)", 'vari_mean': "greenness (VARI)", 'ngrdi_mean': "greenness (NGRDI)",
    'brightness_mean': "brightness", 'brightness_rel_neighbors': "brightness compared with neighbours",
    'dsm_mean': "height", 'dsm_std': "height variation", 'slope_mean': "slope",
    'dsm_rel_neighbors': "height compared with neighbours",
    'hag_mean': "height above ground",
}


def _plain_list(features):
    names = []
    for f in features:
        n = PLAIN_LABELS.get(f, "deep appearance" if f.startswith("deep_") else f)
        if not any(n.split(" (")[0] == m.split(" (")[0] for m in names):   # one "greenness" is enough
            names.append(n)
    names = names[:3]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def pair_verdict(value, top_features):
    """ One plain sentence on why a pair is good / fair / poor. """
    what = _plain_list(top_features)
    rating = jm_rating(value)
    if rating == "good":
        return f"Easy to tell apart, mainly by {what}."
    if rating == "fair":
        return f"Partly alike. They differ mostly in {what}; more varied samples of both would help."
    return (f"Look alike in most respects, so expect mix-ups. Add samples that show where they differ "
            f"(mostly {what}), or merge the two classes.")


def jm_rating(v):
    return "good" if v >= JM_GOOD else ("fair" if v >= JM_FAIR else "poor")


def _jm_fill(v):
    return {"good": _SEP_GOOD, "fair": _SEP_FAIR, "poor": _SEP_POOR}[jm_rating(v)]


def render_separability_image(result, class_colors, focus_pair=None, scale=2):
    """ Draws ObiaEngine.compute_class_separability as one figure: the pairwise
    Jeffries-Matusita table (lower triangle, coloured good / fair / poor), the closest
    pairs with the single features that best separate each, and a signature plot of
    each class's mean per feature, with +-1 SD bands for `focus_pair` (default: the
    closest pair) -- bands for every class at once overlap into an unreadable mass.
    Returns a QImage at `scale`x logical size. """
    classes, feats = result['classes'], result['features']
    jm, n = result['jm'], result['n']
    labels = [FEATURE_LABELS.get(f, f.replace("deep_", "Deep ").replace("_pc", " PC")) for f in feats]

    f_title = _cm_font(20, QFont.Bold, display=True)
    f_sub = _cm_font(13)
    f_head = _cm_font(14, QFont.DemiBold, display=True)
    f_sec = _cm_font(15, QFont.Bold, display=True)
    f_cell = _cm_font(15)
    f_bold = _cm_font(15, QFont.Bold)
    f_small = _cm_font(12)
    f_note = _cm_font(12)
    fm_head, fm_small = QFontMetrics(f_head), QFontMetrics(f_small)

    def color_of(name):
        c = class_colors.get(name)
        return QColor(c) if c is not None else _CM_UNCLASSIFIED

    margin, pad, row_h, head_h, sw = 28, 12, 40, 52, 12
    # JM table: lower triangle; rows = all classes, columns = all but the last, plus a Samples column.
    first_w = max(sw + 8 + fm_head.horizontalAdvance(c) for c in classes) + 2 * pad
    n_w = fm_head.horizontalAdvance("Samples") + 2 * pad
    col_ws = [max(sw + 8 + fm_head.horizontalAdvance(c), 44) + 2 * pad for c in classes[:-1]]
    widths = [first_w, n_w] + col_ws
    table_w = sum(widths)
    table_h = head_h + len(classes) * row_h

    # Closest pairs, lowest JM first (at most 6)
    pairs = sorted(result['jm_feature'], key=lambda k: jm.loc[k[0], k[1]])[:6]
    pair_lines = []
    for a, b in pairs:
        top = list(result['jm_feature'][(a, b)].index[:4])
        v = float(jm.loc[a, b])
        pair_lines.append((a, b, v, pair_verdict(v, top)))
    pairs_h = 30 + len(pair_lines) * 50
    if focus_pair is None and pairs:
        focus_pair = pairs[0]
    focus = set(focus_pair or ())

    # Signature plot geometry
    step, plot_h, legend_h = 44, 260, 48
    plot_w = step * len(feats)
    axis_lab_h = max(fm_small.horizontalAdvance(l) for l in labels) * 0.72 + 18
    sig_h = 30 + legend_h + plot_h + axis_lab_h
    content_w = max(table_w, plot_w + 34, 760)

    notes = [
        f"Separability runs from 0 (identical) to 2 (completely distinct): {JM_GOOD} or more is good, "
        f"{JM_FAIR}–{JM_GOOD} fair, below {JM_FAIR} poor. Cross-validated: each pair is separated along the "
        "best direction found on part of the samples and scored on the rest, so it doesn't saturate at 2 "
        "with many features (the full-dimensional score used up to v1.1 did).",
        "The plot shows the selected pair only, with the most telling features on the left; each feature is "
        "scaled to the pair's combined range. Shaded bands are each class's usual range: the less they "
        "overlap, the easier the two are to tell apart.",
    ]
    few = result.get('few', [])
    if few:
        notes.append("Too few samples for a dependable score (under 30; its pairs can read high or low by "
                     "chance): " + ", ".join(few) + ". Add samples to these classes before trusting their pairs.")
    low = [c for c in result['low_n'] if c not in few]
    if low:
        notes.append("Fewer samples than features, so less reliable: " + ", ".join(low) + ".")
    if result['skipped']:
        notes.append("Left out, fewer than 2 samples: " + ", ".join(result['skipped']) + ".")
    note_text = "\n".join(notes)
    note_h = QFontMetrics(f_note).boundingRect(0, 0, int(content_w), 2000, Qt.TextWordWrap, note_text).height()

    title_h = 58
    img_w = content_w + 2 * margin
    img_h = margin + title_h + table_h + 24 + pairs_h + 18 + sig_h + 14 + note_h + margin
    image = QImage(int(img_w * scale), int(img_h * scale), QImage.Format_ARGB32)
    image.fill(_CM_PAPER)
    p = QPainter(image)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.scale(scale, scale)

    p.setPen(_CM_INK)
    p.setFont(f_title)
    p.drawText(QRectF(margin, margin, content_w, 26), Qt.AlignLeft | Qt.AlignVCenter, "Class Separability")
    p.setPen(_CM_MUTED)
    p.setFont(f_sub)
    p.drawText(QRectF(margin, margin + 28, content_w, 20), Qt.AlignLeft | Qt.AlignVCenter,
               f"Jeffries–Matusita distance, cross-validated · {sum(n.values())} training samples · {len(feats)} features")

    # --- JM table ---
    y0 = margin + title_h
    xs = [margin]
    for w in widths:
        xs.append(xs[-1] + w)

    def cell(i, j, text, font=f_cell, color=_CM_INK, fill=None, align=Qt.AlignRight, swatch=None):
        top = y0 if i == 0 else y0 + head_h + (i - 1) * row_h
        rect = QRectF(xs[j], top, xs[j + 1] - xs[j], head_h if i == 0 else row_h)
        if fill is not None:
            p.fillRect(rect, fill)
        p.setPen(QPen(_CM_RULE, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)
        inner = rect.adjusted(pad, 0, -pad, 0)
        if swatch is not None:
            sw_rect = QRectF(inner.left(), rect.center().y() - sw / 2, sw, sw)
            p.fillRect(sw_rect, swatch)
            p.setPen(QPen(swatch.darker(140), 1))
            p.drawRect(sw_rect)
            inner.setLeft(inner.left() + sw + 8)
        p.setPen(color)
        p.setFont(font)
        p.drawText(inner, align | Qt.AlignVCenter, text)

    cell(0, 1, "Samples", f_head, _CM_MUTED)
    for j, c in enumerate(classes[:-1], start=2):
        cell(0, j, c, f_head, align=Qt.AlignLeft, swatch=color_of(c))
    for i, r in enumerate(classes, start=1):
        cell(i, 0, r, f_head, align=Qt.AlignLeft, swatch=color_of(r))
        cell(i, 1, str(n[r]), color=_CM_MUTED)
        for j, c in enumerate(classes[:i - 1], start=2):
            v = float(jm.loc[r, c])
            cell(i, j, f"{v:.2f}", f_cell if jm_rating(v) == "good" else f_bold, fill=_jm_fill(v))
    p.setPen(QPen(_CM_INK, 1.5))
    p.drawLine(QPointF(xs[0], y0 + head_h), QPointF(xs[-1], y0 + head_h))

    # --- Closest pairs ---
    y = y0 + table_h + 24
    p.setPen(_CM_INK)
    p.setFont(f_sec)
    p.drawText(QRectF(margin, y, content_w, 22), Qt.AlignLeft | Qt.AlignVCenter, "Closest pairs")
    y += 30
    name_w = max(fm_head.horizontalAdvance(c) for c in classes) + sw + 6
    for a, b, v, feats_txt in pair_lines:
        x = margin
        for k, name in enumerate((a, b)):
            p.fillRect(QRectF(x, y + 14 - sw / 2, sw, sw), color_of(name))
            p.setPen(_CM_INK)
            p.setFont(f_head)
            p.drawText(QRectF(x + sw + 6, y, name_w, 28), Qt.AlignLeft | Qt.AlignVCenter, name)
            x += name_w + 12
        badge = QRectF(x, y + 4, 96, 20)
        p.fillRect(badge, _jm_fill(v))
        p.setPen(_CM_INK)
        p.setFont(f_bold)
        p.drawText(badge, Qt.AlignCenter, f"{v:.2f} {jm_rating(v)}")
        p.setPen(_CM_MUTED)
        p.setFont(f_small)
        p.drawText(QRectF(margin + sw + 6, y + 26, content_w - sw - 6, 20), Qt.AlignLeft | Qt.AlignVCenter, feats_txt)
        y += 50

    # --- Signature plot ---
    y += 18
    p.setPen(_CM_INK)
    p.setFont(f_sec)
    p.drawText(QRectF(margin, y, content_w, 22), Qt.AlignLeft | Qt.AlignVCenter,
               "Signature plot" + (f" · {focus_pair[0]} vs {focus_pair[1]}" if focus_pair else ""))
    y += 30
    pair_classes = [c for c in classes if c in focus] or classes[:2]
    if focus_pair:
        key = focus_pair if focus_pair in result['jm_feature'] else (focus_pair[1], focus_pair[0])
        order_feats = list(result['jm_feature'][key].index)
    else:
        order_feats = list(feats)
    fidx = [feats.index(f) for f in order_feats]
    plot_labels = [labels[i] for i in fidx]
    lx = margin
    p.setFont(f_small)
    for c in pair_classes:
        p.fillRect(QRectF(lx, y + 8, 14, 4), color_of(c))
        p.setPen(_CM_INK)
        p.drawText(QRectF(lx + 20, y, 300, 20), Qt.AlignLeft | Qt.AlignVCenter, c)
        lx += 20 + fm_small.horizontalAdvance(c) + 22
    y += legend_h

    plot = QRectF(margin + 34, y, plot_w, plot_h)
    means = result['mean'][order_feats].values.astype(float)
    stds = result['std'][order_feats].values.astype(float)
    rows_pair = [classes.index(c) for c in pair_classes]
    # Scale each feature to the pair's combined usual range (mean +-1 SD of both), so the
    # gap between the lines and the overlap of the bands are on a comparable footing.
    lo = (means[rows_pair] - stds[rows_pair]).min(0)
    hi = (means[rows_pair] + stds[rows_pair]).max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)

    def to_y(v):
        # Clipped to [-0.5, 1.5] so a very wide SD band can't swamp the plot.
        v = min(max(float(v), -0.5), 1.5)
        return plot.bottom() - (v + 0.5) / 2.0 * plot.height()

    def to_x(k):
        return plot.left() + (k + 0.5) * step

    p.setPen(QPen(_CM_RULE, 1))
    for gv in (0.0, 0.5, 1.0):
        p.drawLine(QPointF(plot.left(), to_y(gv)), QPointF(plot.right(), to_y(gv)))
    for k in range(len(feats) + 1):
        p.drawLine(QPointF(plot.left() + k * step, plot.top()), QPointF(plot.left() + k * step, plot.bottom()))
    p.setPen(_CM_MUTED)
    p.setFont(f_small)
    for gv in (0.0, 0.5, 1.0):
        p.drawText(QRectF(margin, to_y(gv) - 9, 28, 18), Qt.AlignRight | Qt.AlignVCenter, f"{gv:g}")
    p.drawText(QRectF(plot.left(), plot.top() - 18, plot.width(), 16), Qt.AlignLeft | Qt.AlignVCenter,
               "most telling  →")
    p.drawText(QRectF(plot.left(), plot.top() - 18, plot.width(), 16), Qt.AlignRight | Qt.AlignVCenter,
               "→  least telling")

    for c in pair_classes:
        ci = classes.index(c)
        col = color_of(c)
        m = (means[ci] - lo) / span
        s = stds[ci] / span
        band = QPolygonF([QPointF(to_x(k), to_y(m[k] + s[k])) for k in range(len(feats))]
                         + [QPointF(to_x(k), to_y(m[k] - s[k])) for k in reversed(range(len(feats)))])
        fill = QColor(col)
        fill.setAlpha(60)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawPolygon(band)
        path = QPainterPath(QPointF(to_x(0), to_y(m[0])))
        for k in range(1, len(feats)):
            path.lineTo(QPointF(to_x(k), to_y(m[k])))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(col.darker(115), 2.4))
        p.drawPath(path)
        p.setBrush(col)
        for k in range(len(feats)):
            p.drawEllipse(QPointF(to_x(k), to_y(m[k])), 3, 3)

    p.setPen(_CM_INK)
    p.setFont(f_small)
    for k, lab in enumerate(plot_labels):
        w = fm_small.horizontalAdvance(lab)
        p.save()
        p.translate(to_x(k) + 4, plot.bottom() + 10)
        p.rotate(-45)
        p.drawText(QRectF(-w, -8, w, 16), Qt.AlignRight | Qt.AlignVCenter, lab)
        p.restore()

    # --- Notes ---
    p.setPen(_CM_MUTED)
    p.setFont(f_note)
    p.drawText(QRectF(margin, y + plot_h + axis_lab_h + 14, content_w, note_h), Qt.AlignLeft | Qt.TextWordWrap, note_text)
    p.end()
    return image


# --- Discover Classes widgets ---
def make_chip_qimage(rgb_chw, segmented_image, seg_id, centroid_rc, half_px, size):
    """ A square image chip around a segment, its outline drawn in cyan. """
    r, c = int(round(centroid_rc[0])), int(round(centroid_rc[1]))
    H, W = segmented_image.shape
    r0, r1, c0, c1 = r - half_px, r + half_px, c - half_px, c + half_px
    rr0, rr1, cc0, cc1 = max(0, r0), min(H, r1), max(0, c0), min(W, c1)
    chip = np.zeros((2 * half_px, 2 * half_px, 3), dtype=np.uint8)
    chip[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = np.moveaxis(rgb_chw[:, rr0:rr1, cc0:cc1], 0, -1)
    m = np.zeros((2 * half_px, 2 * half_px), dtype=bool)
    m[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = segmented_image[rr0:rr1, cc0:cc1] == seg_id
    chip = np.ascontiguousarray(chip)
    img = QImage(chip.data, chip.shape[1], chip.shape[0], 3 * chip.shape[1], QImage.Format_RGB888).copy()
    img = img.scaled(size, size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    # Outline drawn at the tile's own resolution (2 px), so it stays visible however much
    # the window was shrunk to fit the tile.
    idx = (np.arange(size) * (2 * half_px) / size).astype(int)
    ms = m[np.ix_(idx, idx)]
    inner = ms & np.roll(ms, 1, 0) & np.roll(ms, -1, 0) & np.roll(ms, 1, 1) & np.roll(ms, -1, 1)
    edge = ms & ~inner
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
    img = img.convertToFormat(QImage.Format_RGB888)
    out = np.array(img.constBits(), dtype=np.uint8)
    out = out.reshape(size, img.bytesPerLine())[:, :3 * size].reshape(size, size, 3).copy()
    out[edge] = (0, 229, 255)
    out = np.ascontiguousarray(out)
    return QImage(out.data, size, size, 3 * size, QImage.Format_RGB888).copy()


def chip_strip_pixmap(chips, size, gap=3):
    pm = QPixmap(max(1, len(chips) * (size + gap) - gap), size)
    pm.fill(QColor("#1e1e1e"))
    p = QPainter(pm)
    for i, img in enumerate(chips):
        p.drawImage(i * (size + gap), 0, img)
    p.end()
    return pm


_TAG_STYLES = {
    "named": "background:#23402c; color:#a8e0b6;",
    "unnamed": "background:#3a3a3a; color:#c8c8c8;",
    "ignored": "background:#2a2a2a; color:#8a8a8a;",
    "hint": "background:#4a3c1c; color:#f1d58f;",
}


class GroupCard(QFrame):
    """ One group in the Discover list: swatch, key, size, status tag, chip strip, name.
    Click selects it; Ctrl+click toggles it in a multi-selection, Shift+click selects the
    range from the last clicked card (for merging). Emits (key, "plain"|"ctrl"|"shift"). """
    clicked = Signal(str, str)

    def __init__(self, key, parent=None):
        super().__init__(parent)
        self.key = key
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 5, 6, 5)
        lay.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.swatch = QLabel()
        self.swatch.setFixedSize(12, 12)
        self.key_label = QLabel(key)
        self.key_label.setStyleSheet("font-weight:600; color:#f0f0f0;")
        self.size_label = QLabel()
        self.size_label.setStyleSheet("color:#9a9a9a;")
        self.tag = QLabel()
        top.addWidget(self.swatch)
        top.addWidget(self.key_label)
        top.addWidget(self.size_label)
        top.addStretch(1)
        top.addWidget(self.tag)
        lay.addLayout(top)
        self.strip = QLabel()
        lay.addWidget(self.strip)
        self.name_label = QLabel()
        lay.addWidget(self.name_label)
        self.set_selected(False)

    def update_content(self, color, size_text, status, name, tag_text, tag_kind, strip_pixmap):
        self.swatch.setStyleSheet(f"background:{color}; border:1px solid #111;")
        self.size_label.setText(size_text)
        self.tag.setText(tag_text)
        self.tag.setStyleSheet("font-size:11px; padding:1px 6px; border-radius:7px; " + _TAG_STYLES[tag_kind])
        self.tag.setFixedWidth(self.tag.fontMetrics().horizontalAdvance(tag_text) + 16)
        if strip_pixmap is not None:
            self.strip.setPixmap(strip_pixmap)
        if status == "ignored":
            self.name_label.setText("Ignored" + (f" ({name})" if name else ""))
            self.name_label.setStyleSheet("color:#8a8a8a; text-decoration: line-through;")
        else:
            self.name_label.setText(name or "Unnamed")
            self.name_label.setStyleSheet("color:#f0f0f0;" if name else "color:#9a9a9a;")

    def set_selected(self, selected, multi=False):
        if selected:
            border = "#e8604c" if not multi else "#f1d58f"
            self.setStyleSheet(f"GroupCard {{ background:#3a2d24; border:1px solid {border}; border-radius:3px; }}")
        else:
            self.setStyleSheet("GroupCard { background:#2d2d2d; border:1px solid #3d3d3d; border-radius:3px; }")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            mods = event.modifiers()
            kind = "shift" if mods & Qt.ShiftModifier else ("ctrl" if mods & Qt.ControlModifier else "plain")
            self.clicked.emit(self.key, kind)
        super().mouseReleaseEvent(event)


class ChipPickerDialog(QDialog):
    """ A grid of segment chips, each with a tick box and an optional caption, plus a
    target dropdown. Used for Find Similar, moving segments into a group, and What to
    Label Next.
    Without on_apply: exec() == Accepted -> ticked_indices(), target().
    With on_apply(indices, target): the dialog stays open; each Add hands the ticked
    chips to on_apply and removes them from the grid, until the user closes it. """

    def __init__(self, title, intro, chips, captions, targets, target_label, accept_text,
                 default_target=None, ticked=True, columns=5, parent=None, on_apply=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.columns = columns
        self.on_apply = on_apply
        lay = QVBoxLayout(self)
        intro_label = QLabel(intro)
        intro_label.setWordWrap(True)
        lay.addWidget(intro_label)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(8)
        self.boxes, self.cells = [], []
        for i, img in enumerate(chips):
            cell = QVBoxLayout()
            pic = QLabel()
            pic.setPixmap(QPixmap.fromImage(img))
            cell.addWidget(pic)
            box = QCheckBox(captions[i] if captions else "")
            box.setChecked(ticked)
            self.boxes.append(box)
            cell.addWidget(box)
            w = QWidget()
            w.setLayout(cell)
            self.cells.append(w)
        self.alive = list(range(len(chips)))
        self._relayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_host)
        scroll.setMinimumSize(min(columns, max(1, len(chips))) * 150 + 40, 360)
        lay.addWidget(scroll, 1)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#8fbf9a;")
        lay.addWidget(self.status)
        row = QHBoxLayout()
        all_btn = QPushButton("Tick All")
        all_btn.clicked.connect(lambda: [self.boxes[i].setChecked(True) for i in self.alive])
        none_btn = QPushButton("Untick All")
        none_btn.clicked.connect(lambda: [self.boxes[i].setChecked(False) for i in self.alive])
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch(1)
        row.addWidget(QLabel(target_label))
        self.target_combo = QComboBox()
        self.target_combo.addItems(targets)
        if default_target in targets:
            self.target_combo.setCurrentText(default_target)
        row.addWidget(self.target_combo)
        ok = QPushButton(accept_text)
        ok.setStyleSheet("QPushButton { background-color:#c0392b; color:white; font-weight:600; padding:4px 12px; }")
        ok.clicked.connect(self._apply if on_apply else self.accept)
        cancel = QPushButton("Close" if on_apply else "Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addLayout(row)

    def _relayout(self):
        while self.grid.count():
            self.grid.takeAt(0)
        for pos, i in enumerate(self.alive):
            self.grid.addWidget(self.cells[i], pos // self.columns, pos % self.columns)

    def _apply(self):
        picked = self.ticked_indices()
        if not picked:
            self.status.setText("Tick at least one segment first.")
            return
        target = self.target()
        self.on_apply(picked, target)
        for i in picked:
            self.cells[i].hide()
            self.boxes[i].setChecked(False)
        self.alive = [i for i in self.alive if i not in picked]
        self._relayout()
        left = len(self.alive)
        self.status.setText(f"Added {len(picked)} to '{target}'. {left} left" + ("." if left else " -- all done."))

    def ticked_indices(self):
        return [i for i in self.alive if self.boxes[i].isChecked()]

    def target(self):
        return self.target_combo.currentText()


# --- Class colours: suggested from the class name, kept distinct from existing classes ---
_COLOR_KEYWORDS = [
    (("tree", "forest", "canopy", "wood", "crown"), "#1f6b2e"),
    (("palm", "banana", "coconut"), "#2e8b57"),
    (("bamboo",), "#6b8e23"),
    (("brush", "shrub", "bush", "scrub", "thicket", "understor"), "#c9a227"),
    (("grass", "lawn", "pasture", "meadow", "turf", "field"), "#9acd32"),
    (("crop", "rice", "farm", "agri", "corn", "plant"), "#e6d35a"),
    (("bare", "soil", "dirt", "sand", "ground", "earth", "gravel", "clay"), "#a0703c"),
    (("impervious", "road", "pave", "concrete", "asphalt", "parking", "path", "street", "court"), "#7f7f86"),
    (("building", "roof", "house", "structure", "shed", "built"), "#c0392b"),
    (("water", "river", "pond", "lake", "pool", "stream", "creek", "sea"), "#2f80d1"),
    (("shadow", "shade", "dark"), "#3a3a40"),
    (("dead", "snag", "dry", "leafless"), "#b8b0a0"),
    (("cloud", "white"), "#e8e8e8"),
]
_FALLBACK_COLORS = ["#e6194b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#008080",
                    "#9a6324", "#800000", "#000075", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3", "#fb9a99"]


def _color_distance(a, b):
    a, b = QColor(a), QColor(b)
    return ((a.red() - b.red()) ** 2 + (a.green() - b.green()) ** 2 + (a.blue() - b.blue()) ** 2) ** 0.5


def suggest_class_color(name, existing_colors=()):
    """ A colour that matches what the class name describes (trees dark green, grass light
    green, bare brown, roads grey, buildings red, water blue...), shifted lighter or darker
    when an existing class already uses it; names with no match get the palette colour least
    like the existing ones. Returns a hex string. """
    existing = [QColor(c).name() for c in existing_colors]
    low = (name or "").lower()
    base = next((col for words, col in _COLOR_KEYWORDS if any(w in low for w in words)), None)
    if base is not None:
        c = QColor(base)
        for cand in (c, c.lighter(135), c.darker(140), c.lighter(165), c.darker(180), c.lighter(195)):
            if all(_color_distance(cand, e) > 45 for e in existing):
                return cand.name()
    pool = [c for c in _FALLBACK_COLORS if c not in existing] or _FALLBACK_COLORS
    if not existing:
        return pool[0]
    return max(pool, key=lambda c: min(_color_distance(c, e) for e in existing))


def color_icon(color, size=12):
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    p = QPainter(pm)
    p.setPen(QColor("#111111"))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pm)


# --- Lazily filled gallery of segment chips ---
class ChipGallery(QListWidget):
    """ Segment chips that load in batches as you scroll. Grid mode (wrapping, Ctrl/Shift
    multi-select, right-click menu) or strip mode (one horizontal row). Each item keeps its
    segment id in Qt.UserRole. """
    contextRequested = Signal(QPoint)

    def __init__(self, chip_size=96, strip=False, selectable=True, parent=None):
        super().__init__(parent)
        self.chip_size = chip_size
        self._ids, self._chip_fn, self._loaded = [], None, 0
        self._strip = strip
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(chip_size, chip_size))
        self.setGridSize(QSize(chip_size + 6, chip_size + 6))
        self.setMovement(QListWidget.Static)
        self.setResizeMode(QListWidget.Adjust)
        self.setUniformItemSizes(True)
        self.setSpacing(0)
        self.setSelectionMode(QListWidget.ExtendedSelection if selectable else QListWidget.NoSelection)
        self.setStyleSheet("QListWidget { background:#1e1e1e; border:1px solid #3a3a3a; }"
                           "QListWidget::item:selected { background:#e8604c; }")
        if strip:
            self.setFlow(QListWidget.LeftToRight)
            self.setWrapping(False)
            self.setFixedHeight(chip_size + 30)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.horizontalScrollBar().valueChanged.connect(self._maybe_load_more)
        else:
            # Always shown: resizeEvent sizes the columns from the viewport width, and a
            # scrollbar that comes and goes changes that width, which can flip whether the
            # scrollbar is needed -- the grid then flickers between column counts forever.
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.verticalScrollBar().valueChanged.connect(self._maybe_load_more)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: self.contextRequested.emit(self.viewport().mapToGlobal(pos)))

    def set_segments(self, seg_ids, chip_fn, first_batch=48, tip_fn=None):
        """ tip_fn(seg_id) -> tooltip text for a tile (default "Segment #id"). """
        self.clear()
        self._ids, self._chip_fn, self._loaded = list(seg_ids), chip_fn, 0
        self._tip_fn = tip_fn
        self._load(first_batch)
        self.scrollToTop()
        self._scrollbar().setValue(0)

    def _load(self, n):
        end = min(len(self._ids), self._loaded + n)
        for sid in self._ids[self._loaded:end]:
            it = QListWidgetItem(QIcon(QPixmap.fromImage(self._chip_fn(sid, self.chip_size))), "")
            it.setData(Qt.UserRole, int(sid))
            tip_fn = getattr(self, "_tip_fn", None)
            it.setToolTip(tip_fn(int(sid)) if tip_fn else f"Segment #{int(sid)}")
            self.addItem(it)
        self._loaded = end

    def _scrollbar(self):
        return self.horizontalScrollBar() if self._strip else self.verticalScrollBar()

    def _maybe_load_more(self, value):
        bar = self._scrollbar()
        if self._loaded < len(self._ids) and value >= bar.maximum() - self.chip_size:
            self._load(48)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._strip:
            # Grid mode: size the cells so whole columns fill the width (no blank strip).
            avail = self.viewport().width() - 2
            cols = max(1, avail // (self.chip_size + 6))
            cell = max(self.chip_size + 2, avail // cols)
            self.setGridSize(QSize(cell, self.chip_size + 6))
        # Keep filling until the view can scroll, so lazy loading has something to trigger it.
        if self._loaded < len(self._ids) and self._scrollbar().maximum() == 0:
            self._load(48)

    def selected_segment_ids(self):
        return [it.data(Qt.UserRole) for it in self.selectedItems()]


# --- Class name picker: classes (with colour, removable), then Unnamed / Ignored ---
class ClassNameCombo(QComboBox):
    """ Editable dropdown for naming a Discover group. Class entries show their colour and
    a remove cross (click the cross to request removal); the last two entries are Unnamed
    and Ignored. Typing a name that is not a class lets the caller offer an Add button. """
    classChosen = Signal(str)
    specialChosen = Signal(str)      # "unnamed" or "ignored"
    removeRequested = Signal(str)
    renameRequested = Signal(str)
    recolorRequested = Signal(str)
    SPECIAL = {"Unnamed": "unnamed", "Ignored": "ignored"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.lineEdit().setPlaceholderText("Pick a class, or type a new name")
        self.setToolTip("Right-click a class in the list to rename, recolour or remove it.")
        self._classes = []
        self.setItemDelegate(_ClassItemDelegate(self))
        self.view().viewport().installEventFilter(self)
        self.activated.connect(self._on_activated)

    def set_classes(self, classes, current_text):
        """ classes: list of (name, color). """
        self.blockSignals(True)
        self.clear()
        self._classes = [n for n, _ in classes]
        for n, c in classes:
            self.addItem(color_icon(c), n)
        self.insertSeparator(self.count())
        self.addItem("Unnamed")
        self.addItem("Ignored")
        # Show the matching entry (and its colour) only when the text is one; otherwise no
        # entry is current, so no unrelated class colour sits next to a new or empty name.
        self.setCurrentIndex(self.findText(current_text) if current_text else -1)
        self.setEditText(current_text)
        self.blockSignals(False)

    def is_class_row(self, row):
        return 0 <= row < len(self._classes)

    def is_class(self, name):
        return name in self._classes

    def _on_activated(self, row):
        text = self.itemText(row)
        if self.is_class_row(row):
            self.classChosen.emit(text)
        elif text in self.SPECIAL:
            self.specialChosen.emit(self.SPECIAL[text])

    def eventFilter(self, obj, event):
        if obj is self.view().viewport() and event.type() == event.Type.ContextMenu:
            idx = self.view().indexAt(event.pos())
            if idx.isValid() and self.is_class_row(idx.row()):
                name = self._classes[idx.row()]
                from PySide6.QtWidgets import QMenu
                menu = QMenu(self.view())
                act_rename = menu.addAction(f"Rename '{name}'…")
                act_color = menu.addAction("Change Colour…")
                menu.addSeparator()
                act_remove = menu.addAction("Remove Class…")
                chosen = menu.exec(event.globalPos())
                if chosen is not None:
                    self.hidePopup()
                    {act_rename: self.renameRequested, act_color: self.recolorRequested,
                     act_remove: self.removeRequested}[chosen].emit(name)
                return True
        if obj is self.view().viewport() and event.type() == event.Type.MouseButtonRelease:
            idx = self.view().indexAt(event.position().toPoint())
            if idx.isValid() and self.is_class_row(idx.row()):
                rect = self.view().visualRect(idx)
                if event.position().x() >= rect.right() - 24:
                    name = self._classes[idx.row()]
                    self.hidePopup()
                    self.removeRequested.emit(name)
                    return True
        return super().eventFilter(obj, event)


class _ClassItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        combo = self.parent()
        if combo.is_class_row(index.row()):
            r = option.rect
            painter.save()
            painter.setPen(QColor("#e07a6a"))
            f = painter.font()
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(QRectF(r.right() - 22, r.top(), 18, r.height()), Qt.AlignCenter, "✕")
            painter.restore()
