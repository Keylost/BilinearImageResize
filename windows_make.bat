 cd %~dp0
 del /F /Q build
 mkdir build
 cd build
 cmake -G "Visual Studio 14 2015 Win64" ..
 pause
