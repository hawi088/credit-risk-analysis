## Credit Scoring Business Understanding (Expanded & Deeply Referenced)

This section establishes the conceptual foundation for defensible modeling choices in a regulated context. It synthesizes insights from the Basel II Capital Accord, the World Bank's *Credit Scoring Approaches Guidelines* (2019), the HKMA's *Alternative Credit Scoring of MSMEs* white paper (2020), the academic RFMS method (Huang, Zhou, & Wang, 2018), and industry best practices from the Corporate Finance Institute.

---

### Part A: How Basel II's Emphasis on Risk Measurement Demands Interpretable and Well-Documented Models

The Basel II Accord is not merely a capital requirement rulebook; it is a **philosophical framework for risk governance**. Its three pillars collectively create a binding requirement for model interpretability and documentation.

#### Pillar 1: Minimum Capital Requirements – The Direct Link to Model Outputs

Under the Internal Ratings-Based (IRB) approach, banks are permitted to use their own internal estimates of:

- **Probability of Default (PD)** – The likelihood a borrower will default within one year.
- **Loss Given Default (LGD)** – The proportion of exposure lost if default occurs.
- **Exposure at Default (EAD)** – The total exposure at the time of default.

These three parameters directly determine the **risk-weighted assets (RWA)** calculation, which in turn determines how much capital the bank must hold. A model that is not interpretable cannot be reliably validated, and an unvalidated model cannot be used for IRB approval. As the World Bank (2019, p. 5) states: *"The regulatory scrutiny of the management of risk models has intensified around the globe... the guidance explicitly addressed the criticality of strong governance processes."*

**Concrete implication:** If our credit risk model cannot explain *why* a specific customer received a PD of 0.35 versus 0.40, a regulator could reject our IRB application, forcing the bank to use standardized approaches that may require significantly more capital.

#### Pillar 2: Supervisory Review – The Enforcement Mechanism for Documentation

Pillar 2 requires banks to:

1. **Maintain comprehensive model documentation** covering development, validation, implementation, and ongoing monitoring.
2. **Demonstrate conceptual soundness** – the model's theoretical basis must be defensible.
3. **Conduct regular back-testing** – comparing predictions to actual outcomes.
4. **Perform stress testing** – understanding how the model behaves under extreme conditions.

The World Bank (2019, p. 6) notes: *"SR 11-7 highlighted the need to consider 'risk both from individual models and in the aggregate.' Aggregate model risk is affected by interaction and dependencies among models. It is particularly important for credit scoring models given their heavy use in all aspects of the credit life cycle."*

**Concrete implication:** Our proxy variable methodology (RFM clustering to identify "high-risk" customers) must be documented with explicit justifications, including:
- Why we chose 3 clusters (not 2 or 4)
- What validation we performed to confirm the "high-risk" cluster truly represents default-prone behavior
- How we will monitor cluster stability over time

#### Pillar 3: Market Discipline – The "Right to Explain" to Borrowers

Pillar 3 requires banks to disclose their risk management practices to the public. More directly, fair lending laws (ECOA in the US, GDPR in Europe) give borrowers the right to understand why they were denied credit. As the HKMA white paper (2020, p. 50) observes: *"Interpretability has proved a barrier to the adoption of machine learning for the financial industry. If a model is not highly interpretable, a bank may not be permitted to apply its insights to its business."*

**Concrete implication:** A Gradient Boosting model that achieves 0.92 AUC but cannot explain to a rejected applicant *why* they were rejected (beyond "the algorithm said so") exposes the bank to regulatory action and reputational damage. A Logistic Regression model, while potentially less accurate, can provide a clear explanation: "Your loan was denied because your recency score of 45 days (high) and monetary value score of 120 (low) placed you in the high-risk category."

#### Key Regulatory Documents Referenced

| Document | Key Insight for Our Project |
|:---|:---|
| Basel II Pillar 2 (SR 11-7) | Model risk must be managed across the entire lifecycle, from development to retirement. Reproducibility is mandatory. |
| World Bank (2019) – Credit Scoring Approaches Guidelines | Regulators are increasingly concerned about "black box" models. Explainability is not optional; it is a prerequisite for deployment. |
| HKMA (2020) – Alternative Credit Scoring of MSMEs | Feature importance and model interpretability tools (SHAP, LIME) are necessary but not sufficient for regulatory acceptance. |
| CFI – Credit Risk | The 5 Cs of Credit (Character, Capacity, Capital, Collateral, Conditions) remain the industry standard framework. Our model must map to these concepts. |

---

### Part B: Necessity and Risks of Proxy Variable Engineering in the Absence of a Default Label

#### Why a Proxy is Necessary: The RFMS Framework

The raw Xente transaction data contains no column named `is_default`. This is not an oversight; it reflects the reality that default is an *observed outcome* that requires time to manifest. For a new BNPL product, no historical default data exists. Therefore, we must engineer a proxy.

The academic literature provides strong precedent. **Huang, Zhou, and Wang (2018)** in *RFMS Method for Credit Scoring Based on Bank Card Transaction Data* (Statistica Sinica, 28, 2903-2919) faced the same challenge with a Chinese microcredit company. Their solution: extend the classic RFM (Recency, Frequency, Monetary) model from marketing research to credit scoring, adding **S (Standard Deviation)** to capture spending volatility.

Their key findings (Huang et al., 2018, p. 2913-2914):

- **Non-default applicants** tend to have: higher credit-to-debit ratios, higher mean transaction values, more frequent debit card usage, and more total transactions.
- **Default applicants** tend to have: longer recency (time since last transaction), more bank cards owned, more extreme transaction behavior (high maximum values), and longer registration length (suggesting that risk emerges over time).

Their model improved AUC by **13.6%** over the company's basic score (p. 2916), demonstrating that RFM-based features constructed from transactional data have genuine predictive power for credit risk.

#### The Business Risks of Proxy-Based Prediction

Using a proxy introduces risks that must be explicitly disclosed to Bati Bank's leadership and risk committee.

##### Risk 1: Construct Validity (Does the Proxy Measure What We Think It Measures?)

Our proxy will label customers in the lowest RFM segment (low frequency, low monetary value, high recency) as "high risk." But consider:

- **A wealthy customer** who uses a competing platform for most purchases but occasionally uses the eCommerce partner. Low frequency, low monetary value, high recency → labeled high risk. But they are actually low risk due to substantial external assets.
- **A financially distressed customer** who has cut all non-essential spending. Low frequency, low monetary value → labeled high risk. This is correct, but the *reason* is different: they are not "disengaged" but rather "financially constrained."

As the HKMA white paper (2020, p. 30) notes: *"Model fairness: Using the correct data in the machine learning model is crucial... the dataset is often the first place where bias is introduced into a model, and this situation also applies to alternative data."*

**Mitigation:** We will perform cluster profiling, examining not just RFM values but also secondary characteristics (e.g., transaction hour distributions, product categories) to validate that our "high-risk" cluster genuinely exhibits distress signals, not just low engagement.

##### Risk 2: Temporal Instability (Concept Drift)

The relationship between RFM segments and true default risk may change over time. During a recession:

- Previously "good" high-spending customers may default as their income drops.
- Previously "bad" low-spending customers may remain stable (they were already spending minimally).

Our model, trained on pre-recession data, would systematically misclassify.

**Mitigation:** We will implement continuous monitoring with MLflow Model Registry, tracking performance metrics over time and triggering re-training when drift is detected.

##### Risk 3: Selection Bias (The Bank's Existing Customer Base)

The RFM segments are derived from the eCommerce platform's *existing* customers. These customers are already a filtered population (those who have been approved for accounts). As the World Bank (2019, p. 25) warns: *"If historical data are used where social bias was prominent, the algorithm may enforce and amplify the social bias."*

**Mitigation:** We will compare the distribution of our proxy-labeled "high-risk" customers to external benchmarks (industry default rates, if available) and explicitly note this limitation in the final report.

##### Risk 4: Regulatory Scrutiny of Proxy Methodologies

Regulators are increasingly sophisticated about proxy variables. The European Banking Authority (EBA) and the Financial Stability Board (FSB) have both highlighted concerns about *"unintended consequences because the models developed on historical data may learn and perpetuate historical bias"* (World Bank, 2019, p. v).

**Mitigation:** Our documentation will include:
- Explicit justification for the 3-cluster K-Means approach (with random_state fixed for reproducibility)
- Validation analysis showing separation between clusters
- A recommendation to re-train the model once real default data becomes available after 6-12 months of BNPL product operation

---

### Part C: Trade-offs Between Interpretable and High-Performance Models in a Regulated Context

This trade-off is not binary but rather a spectrum. The table below expands significantly on the previous version, incorporating specific insights from all five reference documents.

| Dimension | Logistic Regression with WoE | Gradient Boosting (XGBoost/LightGBM) |
|:---|:---|:---|
| **Interpretability Mechanism** | Coefficients directly map WoE-transformed features to log-odds. The entire model can be expressed as a points-based scorecard (e.g., "500 points = 2% PD"). | Requires post-hoc tools: SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations). These provide *approximations*, not the model's true decision boundary. |
| **Regulatory Precedent** | Well-established. Used by Resona Bank (Japan) for MSME lending (HKMA, 2020, p. 31). Accepted under Basel II IRB approaches. | Evolving. The HKMA (2020, p. 113) notes: *"Model interpretability is a weakness of machine learning algorithms... Emerging techniques like interpretable machine learning have become an increasingly important area of research."* |
| **Performance on Transactional Data** | Moderate. Assumes linear log-odds. May miss interactions (e.g., high frequency + low monetary value = high risk, but each individually is moderate). | High. The CRD Association experiments (HKMA, 2020, p. 78-80) showed XGBoost achieved AUC of 0.8647-0.8694 on full datasets, outperforming Logistic Regression (0.8274-0.8303). |
| **Handling of Non-Linear Patterns** | Requires manual feature engineering (polynomial terms, interaction terms, splines). | Automatic. Decision tree ensembles naturally model thresholds, plateaus, and interactions (e.g., "IF recency > 30 days AND monetary < $500 THEN risk += 0.15"). |
| **Risk of Overfitting** | Low. Simple structure generalizes well. | Moderate to High. Requires careful hyperparameter tuning (max_depth, learning_rate, subsample, colsample_bytree) and cross-validation. The HKMA (2020, p. 46) notes: *"Machine learning algorithms are prone to overfitting... that occurs when the analysis corresponds too closely to a particular set of training data, resulting in a failure to predict future observations accurately."* |
| **Feature Engineering Effort** | High for WoE. Requires binning continuous variables, calculating Information Value (IV) for feature selection, and monotonicity constraints. | Moderate. Can handle raw numerical and categorical features (with encoding). Feature importance can guide feature selection. |
| **Bias Detection** | Transparent. Coefficients reveal which features drive risk. A coefficient on "transaction hour" might reveal bias against night-shift workers. | Opaque. Bias can hide in complex interactions. The World Bank (2019, p. 25) warns: *"A well-intentioned algorithm may inadvertently make biased decisions that may discriminate against protected groups of consumers."* |
| **Computation Time** | Very fast. Minutes to train on 100k rows. | Moderate to High. Hours for large datasets. The CRD experiments (HKMA, 2020, p. 84) note XGBoost required "over five hours" to build the model. |
| **Explainability to Borrowers** | Direct. "Your risk score is high because you have infrequent transactions (low F), low total spend (low M), and haven't purchased in 60+ days (high R)." | Indirect. "SHAP analysis shows your recency value contributed +0.12 to your risk score." This is less intuitive for non-technical borrowers. |

#### The Champion-Challenger Framework in Practice

The HKMA white paper (2020, p. 55-57) recommends the **champion-challenger approach** as the industry standard for managing this trade-off:

> *"The champion–challenger approach involves comparing the results of a conventional credit scoring model (champion) with the results of different alternative credit scoring models (challengers)... Financial lenders can adopt this approach to compare credit score outputs by the existing champion with those by a number of challengers, which are dynamically created by adjusting different rule sets."*

**Our Implementation Plan:**

| Model | Role | Deployment Strategy |
|:---|:---|:---|
| **Logistic Regression + WoE (Champion)** | Primary production model for loan approval decisions. | Deployed in API. Used for all BNPL credit decisions. Fully documented for Basel II compliance. |
| **XGBoost/LightGBM (Challenger)** | Shadow model for performance benchmarking and pattern discovery. | Run in offline mode after each re-training cycle. Results compared to Champion. Features with high importance in Challenger are candidates for engineering into Champion. |
| **Decision Tree (Simplest baseline)** | Interpretability anchor. | Minimal complexity. Used to validate that more complex models are providing genuine value. |

#### The Regulatory Verdict

The World Bank (2019, p. xi) provides the definitive guidance:

> *"CSPs should understand and be able to explain to regulatory bodies the way credit scoring is incorporated into their processes and the logic involved in its functioning. The data used, and the decisions made on the basis of credit scoring, should operate within equal opportunity or anti-discrimination laws... In cases where algorithms are not easily explainable (yet are parsimonious and justifiable), additional steps should be taken to verify that the input data, algorithms, and outputs are performing within expectations."*

**Our Conclusion for Bati Bank:**

Given that this is a first-generation credit risk model for a new BNPL product, operating in a regulated banking environment, **we recommend Logistic Regression with WoE transformation as the primary production model.** This choice:

1. **Meets regulatory requirements** for interpretability and documentation.
2. **Provides a clear audit trail** from input features to output risk score.
3. **Enables straightforward explanation** to rejected applicants, reducing fair lending risk.
4. **Establishes a performance baseline** against which more complex models can be evaluated.

We will train and track Gradient Boosting models as challengers, but they will not serve production loan decisions without additional regulatory review and enhanced explainability tooling (SHAP/LIME integration). This balanced approach maximizes both regulatory safety and technical learning.

---

### References

| Reference | Key Contribution |
|:---|:---|
| Basel Committee on Banking Supervision (2003). *The New Basel Capital Accord (Basel II)*. | Three Pillars: Minimum capital requirements, supervisory review, market discipline. |
| World Bank (2019). *Credit Scoring Approaches Guidelines*. | Regulatory expectations for interpretability, fairness, and model governance. |
| Hong Kong Monetary Authority (2020). *Alternative Credit Scoring of Micro-, Small and Medium-sized Enterprises*. | RFM-based feature engineering, machine learning experiments, champion-challenger framework. |
| Huang, D., Zhou, J., & Wang, H. (2018). RFMS Method for Credit Scoring Based on Bank Card Transaction Data. *Statistica Sinica, 28*, 2903-2919. | Academic validation of RFM-based credit scoring from transactional data. 13.6% AUC improvement over baseline. |
| Corporate Finance Institute. *Credit Risk*. | 5 Cs of Credit framework: Character, Capacity, Capital, Collateral, Conditions. |