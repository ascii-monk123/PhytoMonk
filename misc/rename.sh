#!/bin/bash

#get current directory
cur_dir="$(pwd)"

#get filename
echo "Enter folder name"
read folder_name

printf "\n"

#check if folder is given
if [ -z "$folder_name" ] || [ "$folder_name" = " " ]; 
then
    printf "Err! Directory cannot be empty\n"
    exit 1
fi

#check if the given folder exists in the current directory
if [ ! -d "$cur_dir/$folder_name" ];
then
    printf "Error! Specified directory cannot be found.\n"
    exit 1
fi

#loop and change filenames in the folder if folder exists
cd $folder_name

idx=1;
for FILE in * ; 

do
mv  "$FILE" "corn_$idx.jpg" 
((idx=idx+1))

done;


printf "Processed a total of {$idx} files\n"

