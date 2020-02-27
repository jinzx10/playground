#!/bin/bash

a=(3 6 9)

for i in "${a[@]}"
do
	echo $i
done

for i in {0..2}
do
	echo $i ${a[i]}
done
