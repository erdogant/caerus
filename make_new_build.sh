echo "Making new build.."
echo ""
python setup.py bdist_wheel
echo ""
read -p "Making source build after pressing [Enter].."
echo 
python setup.py bdist_wheel
echo ""
read -p "Press [Enter] to install the pip package..."
pip install -U dist/caerus-0.1.0-py3-none-any.whl
echo ""
read -p "Press [Enter] key to close window..."
echo ""
