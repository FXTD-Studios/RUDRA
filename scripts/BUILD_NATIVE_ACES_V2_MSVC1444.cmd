@echo off
rem Local installed-toolchain fallback: MSVC 19.51 fails current Torch headers.
rem Run from the repository root. Separate Ninja build preserves the failed build.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
set "RUDRA_CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "RUDRA_NINJA=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
"%RUDRA_CMAKE%" -S native -B build/native_aces_v2_msvc1444 -G Ninja "-DCMAKE_MAKE_PROGRAM=%RUDRA_NINJA%" -DCMAKE_BUILD_TYPE=Release -DRUDRA_BUILD_TESTS=ON -DRUDRA_BUILD_CLI=ON -DRUDRA_BUILD_APP=OFF -DRUDRA_WITH_LIBTORCH=ON "-DRUDRA_TORCH_ROOT=%CD%/.venv/Lib/site-packages/torch" -DRUDRA_WITH_OCIO=ON -DRUDRA_WITH_OPENCV=ON "-DOpenCV_DIR=%CD%/tmp/native_deps/opencv-4.10.0/opencv/build/x64/vc16/lib" "-DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows" -DRUDRA_WITH_ONNXRUNTIME=ON "-DONNXRUNTIME_ROOT=%CD%/tmp/native_deps/ort-root-1.22.0"
if errorlevel 1 exit /b 1
"%RUDRA_CMAKE%" --build build/native_aces_v2_msvc1444 --parallel 2 --target rudra_tests rudra-native aces_v2_parity aces_v2_constraint_parity
exit /b %errorlevel%
