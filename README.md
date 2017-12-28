# Adverserial Examples

Playground for experimentation with adverserial examples.

For generation of the adverserial example using boundary attack, execute the script:
```
python boundaryAttack.py --writeImages
```
Passing in the --writeImages flag will write the corresponding adverserial images into the specified log directory.

![Input image](images/ILSVRC2012_val_00038400-input.png?raw=true "Input Image")
![Initial adverserial image (Random uniform)](images/ILSVRC2012_val_00038400-initial.png?raw=true "Initial adverserial image (Random uniform)")
![Adverserial image (First iteration)](images/ILSVRC2012_val_00038400-adverserial-0.png?raw=true "Adverserial image (First iteration)")
![Adverserial image (Second iteration)](images/ILSVRC2012_val_00038400-adverserial-1.png?raw=true "Adverserial image (Second iteration)")
![Adverserial image (Third iteration)](images/ILSVRC2012_val_00038400-adverserial-2.png?raw=true "Adverserial image (Third iteration)")
![Adverserial image (Fourth iteration)](images/ILSVRC2012_val_00038400-adverserial-3.png?raw=true "Adverserial image (Fourth iteration)")
![Adverserial image (Fifth iteration)](images/ILSVRC2012_val_00038400-adverserial-4.png?raw=true "Adverserial image (Fifth iteration)")
![Adverserial image (Sixth iteration)](images/ILSVRC2012_val_00038400-adverserial-5.png?raw=true "Adverserial image (Sixth iteration)")

<h2>References:</h2>
<ol>
<li>https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/boundary_attack.py</li>
</ol>

<br/> Author: <b>Shoaib Ahmed Siddiqui</b>
<br/> Email: <b>12bscsssiddiqui@seecs.edu.pk</b>
