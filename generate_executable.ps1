$Folder = '.\gnr602'
if (Test-Path -Path $Folder) {
    "virtual environment exists!"
} else {
    python -m venv gnr602
}

rm *.spec
rm -r -fo .\build
rm -r -fo .\dist
.\gnr602\Scripts\Activate
pip install -r requirements.txt
pyinstaller -F --paths=".\gnr602\Lib\site-packages" kohonen.py --hidden-import='PIL._tkinter_finder'