if [ -d "gnr602" ]; then
    echo "virtual environment exists"
else
    python3 -m venv gnr602 
fi

rm *.spec
rm -rf ./build
rm -rf ./dist
source gnr602/bin/activate
pip install -r requirements.txt
pyinstaller -F --paths=./gnr602/lib/python3.8/site-packages kohonen.py --hidden-import='PIL._tkinter_finder'