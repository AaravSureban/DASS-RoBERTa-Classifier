<h1 align="center" id="title">DASS-RoBERTa-Classifier</h1>

<p align="center"><img src="https://socialify.git.ci/AaravSureban/DASS-RoBERTa-Classifier/image?custom_description=A+RoBERTa-based+model+I+developed+in+my+research+to+classify+depression%2C+anxiety%2C+and+stress+severity+levels+from+DASS-42+survey+responses.&amp;custom_language=Python&amp;description=1&amp;font=Raleway&amp;language=1&amp;name=1&amp;pattern=Floating+Cogs&amp;theme=Auto" alt="project-image"></p>

<p id="description">
This repository implements a RoBERTa-based approach to automatically assess depression, anxiety, and stress severity from DASS-42 survey responses, a critical step in early mental health screening and intervention. The DASS-42 is a well-validated 42-item questionnaire widely used in clinical and research settings, but manual scoring can be time-consuming and prone to errors. By fine-tuning RoBERTa on thousands of responses, this project delivers an automated classifier that achieves ≥ 90 % accuracy on each subscale.
</p>

- **`model.py`** performs full fine-tuning of RoBERTa-base for 4 epochs on each of the three severity tasks, saving the best validation checkpoints.  
- **`evaluate.py`** loads those checkpoints and runs both a strict 60/20/20 hold-out evaluation and 5-fold stratified cross-validation (with early stopping, dropout, weight decay, and label smoothing) to demonstrate robust generalization and avoid over-fit.

For the full methodology, detailed results, and discussion of implications for mental health care, please see the accompanying paper:    
🔗 [**Transforming Mental Health Care: Harnessing the Power of RoBERTa for DASS-42 Classification**]([Transforming-Mental-Health-Care-Harnessing-the-Power-of-RoBERTa.pdf](https://nhsjs.com/2023/transforming-mental-health-care-harnessing-the-power-of-roberta-for-assessing-and-supporting-anxiety-stress-and-de))


<h2>🛠️ Installation & Setup:</h2>

<p>1. Clone the repository</p>

<p>2. Create a Python virtual environment</p>
<pre><code>python -m venv venv</code></pre>

<p>3. Activate the virtual environment (Windows PowerShell)</p>
<pre><code>.\venv\Scripts\Activate.ps1</code></pre>

<p>4. Install required dependencies</p>
<pre><code>pip install --upgrade pip
pip install -r requirements.txt
</code></pre>


<p>5. Run the full fine-tuning script</p>
<pre><code>python model.py
</code></pre>


<p>6. Evaluate hold-out test & 5-fold cross-validation</p>
<pre><code>python evaluate_generalization.py
</code></pre>


<p>7. Run a quick inference on a dummy example to validate results</p>
<pre><code>python inference_demo.py \
  --model depression_head_finetuned.pt \
  --text "0 1 2 1 0 3 2 1 … 0" \
  --task depression
</code></pre>

<p>8. Tweak hyperparameters (if needed) and re-run</p>
<pre><code># Edit learning rates, batch sizes, epochs, etc. in model.py or evaluate_generalization.py
# Then re-run steps 6 and 7:
python model.py
python evaluate_generalization.py
</code></pre>

<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
<h2>🛡️ License:</h2>

This project is licensed under the MIT License
