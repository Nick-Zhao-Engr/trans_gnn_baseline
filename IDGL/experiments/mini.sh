for i in $(seq 1 4);do
python main.py --config config/mini/idgl_anchor.yml --multi_run
done

# python main.py --config config/mini/idgl_anchor.yml