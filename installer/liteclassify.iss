; Lite Classify installer (Inno Setup 6). One script, two editions:
;   ISCC /DEdition=CPU /DAppVersion=1.0 installer\liteclassify.iss    (from dist\cpu\LiteClassify)
;   ISCC /DEdition=GPU /DAppVersion=1.0 installer\liteclassify.iss    (from dist\gpu\LiteClassify)
; installer\build_installers.py builds both apps and both installers. See installer\README.md.

#ifndef Edition
  #define Edition "CPU"
#endif
#ifndef AppVersion
  #define AppVersion "1.0"
#endif
#if Edition == "GPU"
  #define SourceDir "..\dist\gpu\LiteClassify"
  #define EditionLong "CPU + NVIDIA GPU"
#else
  #define SourceDir "..\dist\cpu\LiteClassify"
  #define EditionLong "CPU only"
#endif

[Setup]
; One AppId for both editions: installing one over the other replaces it.
AppId={{BD3EACC7-2326-4925-BBFB-1C70EF7E8216}
AppName=Lite Classify
AppVersion={#AppVersion}
AppVerName=Lite Classify {#AppVersion} ({#EditionLong})
AppPublisher=LeoBar
AppPublisherURL=https://leobarua.github.io/liteclassify/
AppSupportURL=https://leobarua.github.io/liteclassify/
AppComments=Object Based Image Classification of drone orthomosaics
VersionInfoVersion={#AppVersion}
VersionInfoProductName=Lite Classify
VersionInfoDescription=Lite Classify {#AppVersion} setup ({#EditionLong})
DefaultDirName={autopf}\Lite Classify
DefaultGroupName=Lite Classify
DisableProgramGroupPage=yes
; Per-user install by default (no admin needed); the dialog lets an admin install for all users.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile=..\LICENSE
SetupIconFile=..\logo_liteclassify.ico
UninstallDisplayIcon={app}\LiteClassify.exe
UninstallDisplayName=Lite Classify {#AppVersion}
WizardStyle=modern
WizardImageFile=wizard_large.bmp,wizard_large_200.bmp
WizardSmallImageFile=wizard_small.bmp,wizard_small_200.bmp
OutputDir=..\releases\installers
OutputBaseFilename=LiteClassify-{#AppVersion}-Setup-{#Edition}
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
#ifdef Span
; Only if the compressed data ever exceeds Inno Setup's ~2 GB single-file limit (ISCC /DSpan):
; slices then sit beside the .exe as -1.bin, -2.bin... The GPU edition is ~1.8 GB today.
DiskSpanning=yes
DiskSliceSize=max
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Switching editions (or upgrading) must not leave the other edition's libraries behind.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Lite Classify"; Filename: "{app}\LiteClassify.exe"; Comment: "Object Based Image Classification"
Name: "{autodesktop}\Lite Classify"; Filename: "{app}\LiteClassify.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\LiteClassify.exe"; Description: "{cm:LaunchProgram,Lite Classify}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
#if Edition == "GPU"
  { nvcuda.dll comes with the NVIDIA driver; without it the GPU edition only runs on CPU. }
  if not FileExists(ExpandConstant('{sys}\nvcuda.dll')) then
    Result := MsgBox('No NVIDIA graphics driver was found on this computer.' + #13#10 + #13#10 +
                     'This is the CPU + GPU edition (large download). Without an NVIDIA GPU it runs on the CPU only, ' +
                     'exactly like the much smaller CPU-only edition.' + #13#10 + #13#10 +
                     'Install the GPU edition anyway?', mbConfirmation, MB_YESNO) = IDYES;
#endif
end;
