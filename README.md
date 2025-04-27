## Compile & Run on Windows
```cmd
rem compile (use Visual Studio Developer Command Prompt)
mkdir build
cd build
cmake ..
cmake --build .

rem run
Debug\headless.exe

rem play the video file
ffplay hwenc.264
```
