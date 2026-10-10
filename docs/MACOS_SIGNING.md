# Signing and notarizing the macOS app

Without this, `scripts/package_mac.sh` signs RUDRA.app ad hoc and macOS says
the app "cannot be checked for malicious software" on first open. With a
Developer ID certificate and a notary key, the script signs every binary with
the hardened runtime, has Apple notarize the app and the DMG, staples both
tickets and checks the result with Gatekeeper, so the DMG opens with no warning,
offline too.

## One time, on the Apple side (the account holder)

1. **Enroll** in the Apple Developer Program (developer.apple.com/programs,
   99 USD a year). Enroll as an organization (FXTD Studios) so the certificate
   reads "Developer ID Application: FXTD Studios (TEAMID)"; that needs a
   D-U-N-S number. An individual enrollment works too, under your own name.
2. **Developer ID Application certificate.** On a Mac: Xcode > Settings >
   Accounts > Manage Certificates > + > Developer ID Application. Or
   developer.apple.com > Certificates > + > Developer ID Application with a CSR
   from Keychain Access. Only the Account Holder can create it.
3. **Export it** from Keychain Access (My Certificates, the certificate with
   its private key) as `RUDRA-devid.p12` with a password.
4. **Notary API key.** appstoreconnect.apple.com > Users and Access >
   Integrations > App Store Connect API > Team Keys > + with the Developer
   role. Download `AuthKey_<KEYID>.p8` (once only) and note the Key ID and the
   Issuer ID shown above the list.

## The GitHub secrets (repository > Settings > Secrets and variables > Actions)

| Secret | Value |
|---|---|
| `MACOS_CERT_P12` | `base64 -i RUDRA-devid.p12` (one line) |
| `MACOS_CERT_PASSWORD` | the export password |
| `MACOS_SIGN_IDENTITY` | `Developer ID Application: FXTD Studios (TEAMID)`, exactly as `security find-identity -v -p codesigning` prints it |
| `MACOS_NOTARY_KEY` | `base64 -i AuthKey_<KEYID>.p8` |
| `MACOS_NOTARY_KEY_ID` | the Key ID |
| `MACOS_NOTARY_ISSUER` | the Issuer ID |

`.github/workflows/release.yml` imports the certificate into a throwaway
keychain, passes the rest to the script and deletes both afterwards. With no
`MACOS_CERT_P12` the release still builds, signed ad hoc, with a warning in the
run. The run summary says which way the DMG was signed.

## On a Mac by hand

```bash
# the certificate in your login keychain, then either the API key ...
MAC_SIGN_IDENTITY="Developer ID Application: FXTD Studios (TEAMID)" \
MAC_NOTARY_KEY=~/keys/AuthKey_XXXX.p8 MAC_NOTARY_KEY_ID=XXXX MAC_NOTARY_ISSUER=<issuer uuid> \
  scripts/package_mac.sh
# ... or a stored profile: xcrun notarytool store-credentials rudra --key ... --key-id ... --issuer ...
MAC_SIGN_IDENTITY="..." MAC_NOTARY_PROFILE=rudra scripts/package_mac.sh
```

A signed build without notarization stops with an error, because it would
still warn; `ALLOW_UNNOTARIZED=1` keeps it for a local test. If Apple rejects a
submission the script prints the notary log, which names the binary at fault.

## Checking a DMG

```bash
spctl --assess --type open --context context:primary-signature -vv RUDRA-*.dmg   # source=Notarized Developer ID
xcrun stapler validate RUDRA-*.dmg
```
