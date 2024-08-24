@echo off
REM Crear un entorno virtual
virtualenv -p python3 env

REM Activar el entorno virtual
call .\env\Scripts\activate

REM Instalar las dependencias desde requirements.txt
pip install --no-cache-dir -r requirements.txt

REM Ejecutar el script Python
python .\src\app.py

REM Mensaje de finalización
echo Configuracion completa
pause

