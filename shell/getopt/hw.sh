#!/bin/bash

args=$( getopt --options=q:t: --longoptions=queue:,time: -- "$@" )

eval set -- $args

while true; do
	case "$1" in
		-q|--queue)
			iq="$2"; shift 2 ;;
		-t|--time)
			it="$2"; shift 2 ;;
		--)
			shift; break;;
	esac
done

echo $iq
echo $it
