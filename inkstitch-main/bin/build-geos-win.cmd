@echo off
mkdir %CD%\geos-build
SET GEOS_INSTALL=%CD%\geos-build

if NOT DEFINED BUILD32 (SET BUILDFLAG="x64") else (SET BUILDFLAG="Win32")
echo %BUILDFLAG%

curl -L -O  https://github.com/libgeos/geos/releases/download/3.12.2/geos-3.12.2.tar.bz2

7z x geos-3.12.2.tar.bz2
7z x geos-3.12.2.tar

move geos-3.12.2 geos
cd geos
cmake -S . -B _build -G "Visual Studio 16 2019" -A %BUILDFLAG% -DCMAKE_INSTALL_PREFIX=%GEOS_INSTALL% -DCMAKE_GENERATOR_TOOLSET=host=x64 -DBUILD_TESTING=OFF

cmake --build _build --config Release -j 16 --verbose
cd _build
cmake --install .

SET PATH=%GEOS_INSTALL%\bin;%PATH%
SET GEOS_INCLUDE_PATH=%GEOS_INSTALL%\include
set GEOS_LIBRARY_PATH=%GEOS_INSTALL%\lib

xcopy %GEOS_INSTALL%\bin\geos_c.dll %pythonLocation% /E /H /C /I
xcopy %GEOS_INSTALL%\bin\geos.dll %pythonLocation% /E /H /C /I

python -m pip uninstall -y shapely
python -m pip cache remove shapely
python -m pip install -v shapely --no-binary shapely
