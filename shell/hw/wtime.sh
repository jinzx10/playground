#!/bin/bash

t1=00:30:00
t2=20:00:00

if [[ "$t1" > "$t2" ]]; then 
	echo "$t1 > $t2"
else
	echo "$t1 < $t2"
fi
