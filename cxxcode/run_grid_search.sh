for embedding in 32 48 64 128 256
do
  for hidden in 32 48 64 128 256
  do
    condorizer --output 2015_sent-lr1_$embedding"_"$hidden dtc --task train --trnfile /scratch/cluster/elisaf/cs395t/finalProject/data/yelp_2015/output/trn-yelp.txt --devfile /scratch/cluster/elisaf/cs395t/finalProject/data/yelp_2015/output/dev-yelp.txt --nclass=5 --path /scratch/cluster/elisaf/cs395t/finalProject/data/yelp_2015/models/ --verbose=1 --ndisrela 36 --droprate=0.3 --inputdim=$embedding --hiddendim=$hidden --lr=0.1 --trainer=1 --niter=1
  done
done
