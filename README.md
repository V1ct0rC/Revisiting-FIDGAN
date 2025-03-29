# Revisiting-FIDGAN

Reimplementation of the FIDGAN architecture for network intrusion detection. 

This implementation is based on the paper "P. Freitas de Araujo-Filho, G. Kaddoum, D. R. Campelo, A. Gondim Santos, D. Macêdo and C. Zanchettin, "**Intrusion Detection for Cyber–Physical Systems Using Generative Adversarial Networks in Fog Environment**," in IEEE Internet of Things Journal, vol. 8, no. 8, pp. 6247-6256, 15 April15, 2021, doi: 10.1109/JIOT.2020.3024800." and is made for the course IF848 Detecção de Intrusão.

To run the code, create a conda environment with the following command:

```bash
conda env create -f environment.yml
```

Then, activate the environment with:
```bash
conda activate fidgan
```

And run the main script with the desired dataset (replace <> with the dataset name, 'wadi' or 'CICIDS'):
```bash
python main.py --settings datasets\<>\<>_settings.json
```

You can see the original code at https://github.com/pfreitasaf/FIDGAN