Here is a simple explanation of how the Layout Conductor works:

* Remove TABLES/FIGURES/CAPTIONS/FOOTNOTES from the reading order(since these elements do not have any reading order)
* Cluster the remaining text boxes to produce merged bbox's for the final layout which can be read directly

To find the tables/figures/captions we use multiple models, since they are individually not super accurate:
* PDFFIGURES by AllenAI [https://github.com/allenai/pdffigures2]
* Layout Parser EFFDET [http://layout-parser.github.io/]
* VILA token preds [https://github.com/allenai/VILA]

We reconcile the predictions from these separate models to aggragate all sources of tables/figures/captions/ancillary text and then cluster the remaining text using Hierarchical Clustering[https://en.wikipedia.org/wiki/Hierarchical_clustering].

We also include a simple image annotation gui tool, which helps evaluate the results quickly on a scale of 1-3.

## Lets start with the dependency installation

Lets assume your pdfs are stored in path : `./docs/`

### First lets install the PDFFigures library(and scala) by AllenAI before continuing running the library:
```
! git clone https://github.com/SwapnilDreams100/pdffigures2
! cd ./pdffigures2
! sudo apt-get update
! sudo apt-get install apt-transport-https curl gnupg -yqq
! echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
! echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
! curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo -H gpg --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/scalasbt-release.gpg --import
! sudo chmod 644 /etc/apt/trusted.gpg.d/scalasbt-release.gpg
! sudo apt-get update
! sudo apt-get install sbt
```

Above installation which works on Google colab/Ubuntu, pl refer to the library for specific instrucions for other distros/os : https://github.com/allenai/pdffigures2

Then we can run the followijng command from within the folder to process these pdfs and extract some of its figures into folder `./fig_outs`:

`! sbt "runMain org.allenai.pdffigures2.FigureExtractorBatchCli ../docs -s stat_file.json -m ./ -d ../fig_outs/"`

### Next lets install the VILA library by AllenAI and its corresponding dependencies:
```
! git clone https://github.com/allenai/VILA
! cd ./VILA
! pip install -e .
! pip install -r requirements.txt
! apt-get install poppler-utils 
! pip install sklearn tqdm
```

Thats it! Now we can run the LayoutConductor module and produce the correct reading order!
Pl refer to the `LayoutConductorTutorial.ipynb` for example code snippets!

### To run the annotator, an additional dependency of PyQt5 is required:

```
! pip install pyqt5
! python annotate_gui.py
```

Thanks for using this library!
