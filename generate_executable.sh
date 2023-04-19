rm *.spec
rm -rf ./build
rm -rf ./dist
pyinstaller -F --paths=~/Desktop/kohonen-self-organising-map/gnr602/lib/python3.8/site-packages kohonen.py --hidden-import='PIL._tkinter_finder'