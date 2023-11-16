#!/bin/bash

topdir=/home/zuxin/playground/python/orbgen/In/
bond_length=(3.00 3.20 3.40 3.60)

#pp_class="pd04"
#pp_file=In.PD04.PBE.UPF

pp_class="sg15v1.0"
pp_file="In_ONCV_PBE-1.0.upf"

orb_file="In_opt_7au_100Ry_1s1p1d.orb"

for len in ${bond_length[@]}; do
    cd ${topdir}
    jobdir=${topdir}/${pp_class}/${len}

    mkdir -p ${jobdir}
    cp INPUT STRU KPT ${jobdir}
    cd ${jobdir}

    sed -e "s/LENGTH/${len}/" \
        -e "s/PP_FILE/${pp_file}/" \
        -e "s/ORB_FILE/${orb_file}/" \
        -i STRU

    ln -sf ${topdir}/pp/${pp_file} ${jobdir}/${pp_file}
    ln -sf ${topdir}/orb/${orb_file} ${jobdir}/${orb_file}
done
