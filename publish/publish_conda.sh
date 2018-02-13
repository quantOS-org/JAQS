conda skeleton pypi jaqs
cp build1.sh jaqs
grep -rl "^.*enum34.*$" jaqs | xargs sed -i "s|^.*enum34.*$||g"
conda config --set anaconda_upload yes
conda-build jaqs
