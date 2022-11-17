gamma=10
final_gamma=101
for((;gamma<final_gamma;gamma=gamma+1))
do
python clouding.py $gamma
done
