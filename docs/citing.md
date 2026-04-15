# Citing JaQMC

If you use JaQMC in your research, please cite the following paper, which introduced the first version of the software.

> W. Ren, W. Fu, X. Wu, and J. Chen, Towards the ground state of molecules via diffusion Monte Carlo on neural networks, Nat Commun 14, 1 (2023).

````{dropdown} BibTeX entry
```bibtex
@article{ren_towards_2023,
  title = {Towards the Ground State of Molecules via Diffusion {{Monte Carlo}} on Neural Networks},
  author = {Ren, Weiluo and Fu, Weizhong and Wu, Xiaojie and Chen, Ji},
  year = 2023,
  month = apr,
  journal = {Nature Communications},
  volume = {14},
  number = {1},
  pages = {1860},
  publisher = {Nature Publishing Group},
  issn = {2041-1723},
  doi = {10.1038/s41467-023-37609-3},
}
```
````

If you use any of the following techniques, please also cite the corresponding paper or papers.

````{dropdown} FermiNet architecture (enabled by default)
```bibtex
@article{pfau_ferminet_2020,
  title = {Ab Initio Solution of the Many-Electron {{Schr\"odinger}} Equation with Deep Neural Networks},
  author = {Pfau, David and Spencer, James S. and Matthews, Alexander G. D. G. and Foulkes, W. M. C.},
  year = 2020,
  month = sep,
  journal = {Physical Review Research},
  volume = {2},
  number = {3},
  pages = {033429},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.2.033429},
}
```
````

````{dropdown} Psiformer architecture
```bibtex
@inproceedings{glehn_psiformer_2023,
  title = {A Self-Attention Ansatz for Ab-Initio Quantum Chemistry},
  booktitle = {The Eleventh International Conference on Learning Representations, {{ICLR}} 2023},
  author = {{von Glehn}, Ingrid and Spencer, James S. and Pfau, David},
  year = 2023,
  publisher = {OpenReview.net},
  address = {kigali, rwanda},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/conf/iclr/GlehnSP23.bib},
}
```
````

````{dropdown} Forward Laplacian (enabled by default)
```bibtex
@article{li_fwdlap_2024,
  title = {A Computational Framework for Neural Network-Based Variational {{Monte Carlo}} with {{Forward Laplacian}}},
  author = {Li, Ruichen and Ye, Haotian and Jiang, Du and Wen, Xuelan and Wang, Chuwei and Li, Zhe and Li, Xiang and He, Di and Chen, Ji and Ren, Weiluo and Wang, Liwei},
  year = 2024,
  month = feb,
  journal = {Nature Machine Intelligence},
  volume = {6},
  number = {2},
  pages = {209--219},
  publisher = {Nature Publishing Group},
  issn = {2522-5839},
  doi = {10.1038/s42256-024-00794-x},
}
```
````

````{dropdown} Stochastic reconfiguration (SR)
```bibtex
@article{chen_min-SR_2024,
  title = {Empowering Deep Neural Quantum States through Efficient Optimization},
  author = {Chen, Ao and Heyl, Markus},
  year = 2024,
  month = jul,
  journal = {Nature Physics},
  volume = {20},
  pages = {1476--1481},
  publisher = {Nature Publishing Group},
  issn = {1745-2481},
  doi = {10.1038/s41567-024-02566-1},
}
```
If you enabled `spring_mu`, also cite:
```bibtex
@article{goldshlager_spring_2024,
  title = {A {{Kaczmarz-inspired}} Approach to Accelerate the Optimization of Neural Network Wavefunctions},
  author = {Goldshlager, Gil and Abrahamsen, Nilin and Lin, Lin},
  year = 2024,
  month = nov,
  journal = {Journal of Computational Physics},
  volume = {516},
  pages = {113351},
  issn = {0021-9991},
  doi = {10.1016/j.jcp.2024.113351}
}
```
If you enabled `march_beta`, also cite:
```bibtex
@misc{gu_solving_2025,
  title = {Solving the {{Hubbard}} Model with {{Neural Quantum States}}},
  author = {Gu, Yuntian and Li, Wenrui and Lin, Heng and Zhan, Bo and Li, Ruichen and Huang, Yifei and He, Di and Wu, Yantao and Xiang, Tao and Qin, Mingpu and Wang, Liwei and Lv, Dingshun},
  year = 2025,
  month = jul,
  number = {arXiv:2507.02644},
  eprint = {2507.02644},
  primaryclass = {cond-mat},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2507.02644},
  archiveprefix = {arXiv}
}
```
````

````{dropdown} Pseudopotential/ECP
```bibtex
@article{li_fermionic_2022,
  title = {Fermionic Neural Network with Effective Core Potential},
  author = {Li, Xiang and Fan, Cunwei and Ren, Weiluo and Chen, Ji},
  year = 2022,
  month = jan,
  journal = {Physical Review Research},
  volume = {4},
  number = {1},
  pages = {013021},
  issn = {2643-1564},
  doi = {10.1103/PhysRevResearch.4.013021},
}
```
````

````{dropdown} Pseudo Hamiltonian
```bibtex
@misc{fu_local_2025,
  title = {Local {{Pseudopotential Unlocks}} the {{True Potential}} of {{Neural Network-based Quantum Monte Carlo}}},
  author = {Fu, Weizhong and Fujimaru, Ryunosuke and Li, Ruichen and Liu, Yuzhi and Wen, Xuelan and Li, Xiang and Hongo, Kenta and Wang, Liwei and Ichibha, Tom and Maezono, Ryo and Chen, Ji and Ren, Weiluo},
  year = 2025,
  month = may,
  number = {arXiv:2505.19909},
  eprint = {2505.19909},
  primaryclass = {physics},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2505.19909},
  archiveprefix = {arXiv},
}
```
````

````{dropdown} Spin penalty
```bibtex
@article{li_spin-spenalty_2024,
  title = {Spin-Symmetry-Enforced Solution of the Many-Body {{Schr\"odinger}} Equation with a Deep Neural Network},
  author = {Li, Zhe and Lu, Zixiang and Li, Ruichen and Wen, Xuelan and Li, Xiang and Wang, Liwei and Chen, Ji and Ren, Weiluo},
  year = 2024,
  month = dec,
  journal = {Nature Computational Science},
  volume = {4},
  number = {12},
  pages = {910--919},
  publisher = {Nature Publishing Group},
  issn = {2662-8457},
  doi = {10.1038/s43588-024-00730-4},
}
```
````

````{dropdown} Solids
```bibtex
@article{li_deepsolid_2022,
  title = {Ab Initio Calculation of Real Solids via Neural Network Ansatz},
  author = {Li, Xiang and Li, Zhe and Chen, Ji},
  year = 2022,
  month = dec,
  journal = {Nature Communications},
  volume = {13},
  number = {1},
  pages = {7895},
  publisher = {Nature Publishing Group},
  issn = {2041-1723},
  doi = {10.1038/s41467-022-35627-1},
}
```
````

````{dropdown} Fractional quantum Hall
```bibtex
@article{qian_deephall_2025,
  title = {Describing {{Landau Level Mixing}} in {{Fractional Quantum Hall States}} with {{Deep Learning}}},
  author = {Qian, Yubing and Zhao, Tongzhou and Zhang, Jianxiao and Xiang, Tao and Li, Xiang and Chen, Ji},
  year = 2025,
  month = apr,
  journal = {Physical Review Letters},
  volume = {134},
  number = {17},
  pages = {176503},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.134.176503}
}
```
````
