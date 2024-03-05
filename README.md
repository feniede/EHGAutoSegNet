# EHGAutoSegNet

Source code of the paper:

## Automatic Semantic Segmentation of EHG Recordings by Deep Learning: an Approach to a Screening Tool for Use in Clinical Practice

**Authors:**
- Félix Nieto del Amor<sup>1</sup> (feniede@ci2b.upv.es)
- Yiyao Ye Lin<sup>1,4</sup> (yiye@ci2b.upv.es)
- Rogelio Monfort Ortiz<sup>2</sup> (monfort_isaort@gva.es)
- Vicente Jose Diago-Almela<sup>2</sup> (diago_vicalm@gva.es)
- Fernando Modrego Pardo<sup>2</sup> (modrego_ferpar@gva.es)
- Jose L. Martinez-de-Juan<sup>1,4</sup> (jlmartinez@eln.upv.es)
- Dongmei Hao<sup>3,4</sup> (haodongmei@bjut.edu.cn)
- Gema Prats Boluda<sup>1,4</sup> (gprats@ci2b.upv.es)

**Affiliations:**
1. Centro de Investigación e Innovación en Bioingeniería, Universitat Politècnica de València (Ci2B), 46022 Valencia, Spain
2. Servicio de Obstetricia, H.U.P. La Fe, 46026 Valencia, Spain
3. Faculty of Environment and Life, Beijing University of Technology, Beijing International Science and Technology Cooperation Base for Intelligent Physiological Measurement and Clinical Transformation, Beijing 100124, China
4. BJUT-UPV Joint Research Laboratory in Biomedical Engineering

**Abstract:**

**Background and Objective**: Preterm delivery is an important factor in the disease burden of the newborn and infants worldwide. Electrohysterography (EHG) has become a promising technique for predicting this condition, thanks to its high degree of sensitivity. Despite the technological progress made in predicting preterm labor, its use in clinical practice is still limited, one of the main barriers being the lack of tools for automatic signal processing without expert supervision, i.e. automatic screening of motion and respiratory artifacts in EHG records. Our main objective was thus to design and validate an automatic system of segmenting and screening the physiological segments of uterine origin in EHG records for robust characterization of uterine myoelectric activity, predicting preterm labor and help to promote the transferability of the EHG technique to clinical practice. **Methods**: For this, we combined 300 EHG recordings from the TPEHG DS database and 69 EHG recordings from our own database (Ci2B-La Fe) of women with singleton gestations. This dataset was used to train and evaluate U-Net, U-Net++, and U-Net 3+ for semantic segmentation of the physiological and artifacted segments of EHG signals. The model’s predictions were then fine-tuned by post-processing. **Results**: U-Net 3+ outperformed the other models, achieving an area under the ROC curve of 91.4% and an average precision of 96.4% in detecting physiological activity. Thresholds from 0.6 to 0.8 achieved precision from 93.7% to 97.4% and specificity from 81.7% to 94.5%, detecting high-quality physiological segments while maintaining a trade-off between recall and specificity. Post-processing improved the model’s adaptability by fine-tuning both the physiological and corrupted segments, ensuring accurate artifact detection while maintaining physiological segment integrity in EHG signals. **Conclusions**: As automatic segmentation proved to be as effective as double-blind manual segmentation in predicting preterm labor, this automatic segmentation tool fills a crucial gap in the existing preterm delivery prediction system workflow by eliminating the need for double-blind segmentation by experts and facilitates the practical clinical use of EHG. This work potentially contributes to the early detection of authentic preterm labor women and will allow clinicians to design individual patient strategies for maternal health surveillance systems and predict adverse pregnancy outcomes.


## Clean Install from Scratch

The code requires Python (3.5+). The following code lines install the requirements:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run jupyter notebook. Then, run the example file:

```bash
jupyter notebook
```

## Installation in an existing python environment

The package can be installed with the following command:

```bash
pip install EHGAutoSegNet-main.zip
```

To use the package withing python:

```bash
import ehgautosegnet
```

License
=======
    Copyright 2013 Mir Ikram Uddin

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.