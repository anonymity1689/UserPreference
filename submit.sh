#!/bin/tcsh

#PBS -q SINGLE
#PBS -oe 
#PBS -N tw_20
#PBS -l select=1


setenv PATH ${PBS_O_PATH}

cd ${PBS_O_WORKDIR}
../../.././anaconda3/bin/python user_preferences.py 'twitter_1196' 'glove-twitter-50' 'cpu' 3 0 0