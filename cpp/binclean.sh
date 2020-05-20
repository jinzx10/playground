#!/bin/bash

dir=$(dirname $0)
path=$(realpath $dir)

files=$(find $path -type f -executable -exec file '{}' \; | grep 'ELF' | awk -F ":" '{print $1}')

if [ -z "$files" ]; then
	echo "no binary executables found."
	exit 0
fi

echo -e "\nthe following binary executables will be removed: \n"
echo -e "$files\n"
read -p "Proceed? [Y/n] " ans
if [ -z "$ans" ]; then
	ans=y
fi

case "$ans" in
	y|Y|yes)
		echo $files | xargs rm
		echo -e "\nfiles have been removed.\n"
		;;
	n|N|no)
		echo -e "\nfiles are not removed.\n"
		;;
	*)
		echo -e "\ninvalid input.\n"
		;;
esac
