"""One command to rebuild everything: python -m src.build_all"""
from . import baseline, generate_data, generate_inputs

if __name__ == "__main__":
    generate_data.main()
    generate_inputs.main()
    baseline.main()
