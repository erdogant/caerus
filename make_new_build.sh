echo "Making new build.."
echo ""
python setup.py bdist_wheel
echo ""
read -p "Making source build after pressing [Enter].."
echo 
python setup.py bdist_wheel
echo ""
read -p "Press [Enter] key to close window..."