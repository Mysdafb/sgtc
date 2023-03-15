# Usage

clone this repository, and ```cd``` to the code folder

```bash
git clone https://github.com/Mysdafb/sgtc.git
```

create a virtual environment

```bash
python3 -m venv .venv
```

Activate your virtual environment

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

To run an experiment, define the parameter within the _configs.yml_ file, then run the following command:

```bash
python run.py --config-file configs.yml
```
