#!/bin/bash
source L1-Skeleton/venv/bin/activate

for filename in /home/katherine/Documents/Berry4D/petiole_instances/*; do    


    if [ -d "${filename}" ]; then
        continue
    fi

    mkdir Results/l1_medial/
    mkdir Results/l1_medial/raw

    echo "${filename}" 

    name=$(basename "$filename")         
    cd L1-Skeleton
    ./pointcloudl1.sh "${filename}" ../Results/l1_medial/raw/ my_config.json   

    cd ..     
    mv Results/l1_medial/raw/skeleton.skel Results/l1_medial/raw/"${name%.*}".skel 
  
done

python parse_l1.py