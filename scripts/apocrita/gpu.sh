qlogin -pe smp 8 -l gpu=1 -l h_rt=1:0:0 -l h_vmem=8G -l rocky
: <<'END'
source /data/home/ec23709/project/ReID/reid/bin/activate
cd /data/home/ec23709/project/FEL
END