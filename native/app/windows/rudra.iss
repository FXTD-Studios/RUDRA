; The RUDRA installer for Windows x64 (Inno Setup 6). scripts/PACKAGE_WINDOWS.ps1
; compiles it from the packaged folder:
;   ISCC /DAppVersion=0.9.0-beta.1 /DSourceDir=<dist\beta\RUDRA-...-windows-x64> /DOutputDir=<dist\beta> rudra.iss
; Per-user by default (no administrator prompt); "Install for all users" is
; offered and then needs one.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #error SourceDir (the packaged folder) is required
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif

[Setup]
#ifdef ModelFreeEvaluation
AppId={{AE1F58CD-4191-44CD-BEE3-F9B821944058}
AppName=RUDRA Analytic Evaluation
AppVerName=RUDRA Analytic Evaluation {#AppVersion}
DefaultDirName={localappdata}\Programs\RUDRA Analytic Evaluation
DefaultGroupName=RUDRA Analytic Evaluation
UsePreviousAppDir=no
#else
AppId={{6F1E2B7A-3C4D-4E5F-9A8B-52554452410A}
AppName=RUDRA
AppVerName=RUDRA {#AppVersion}
DefaultDirName={autopf}\RUDRA
DefaultGroupName=RUDRA
#endif
AppVersion={#AppVersion}
AppPublisher=FXTD Studios
AppPublisherURL=https://fxtdstudios.com
AppSupportURL=https://github.com/FXTD-Studios/RUDRA/issues
AppCopyright=Copyright 2026 FXTD Studios. PolyForm Noncommercial 1.0.0.
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
LicenseFile={#SourceDir}\LICENSE
SetupIconFile=..\icons\rudra.ico
UninstallDisplayIcon={app}\RUDRA.exe
#ifdef ModelFreeEvaluation
UninstallDisplayName=RUDRA Analytic Evaluation {#AppVersion}
#else
UninstallDisplayName=RUDRA {#AppVersion}
#endif
OutputDir={#OutputDir}
OutputBaseFilename=RUDRA-{#AppVersion}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; .rudra projects open in RUDRA (a double-click, Open with).
#ifndef ModelFreeEvaluation
ChangesAssociations=yes
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

#ifndef ModelFreeEvaluation
[Registry]
Root: HKA; Subkey: "Software\Classes\.rudra"; ValueType: string; ValueName: ""; ValueData: "RUDRA.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\RUDRA.Project"; ValueType: string; ValueName: ""; ValueData: "RUDRA project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\RUDRA.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\RUDRA.exe,0"
Root: HKA; Subkey: "Software\Classes\RUDRA.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\RUDRA.exe"" ""%1"""
#endif

[Icons]
#ifdef ModelFreeEvaluation
Name: "{autoprograms}\RUDRA Analytic Evaluation"; Filename: "{app}\RUDRA.exe"
Name: "{autodesktop}\RUDRA Analytic Evaluation"; Filename: "{app}\RUDRA.exe"; Tasks: desktopicon
#else
Name: "{autoprograms}\RUDRA"; Filename: "{app}\RUDRA.exe"
Name: "{autodesktop}\RUDRA"; Filename: "{app}\RUDRA.exe"; Tasks: desktopicon
#endif

[Run]
Filename: "{app}\RUDRA.exe"; Description: "{cm:LaunchProgram,RUDRA}"; Flags: nowait postinstall skipifsilent
