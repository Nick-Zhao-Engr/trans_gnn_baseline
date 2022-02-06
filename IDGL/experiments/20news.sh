for i in $(seq 1 4);do
python main.py --config config/20news10/idgl_anchor.yml --multi_run
done

# python main.py --config config/20news10/idgl_anchor.yml --multi_run