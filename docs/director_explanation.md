# RetentionIQ — The Story Behind the Notebooks

## Part 1: The Full Educational Explanation (Notebook by Notebook)

---

### Chapter 0 — Why Any of This Exists

Before a single line of code, there is a business problem. Miracle Sheets sells
high-quality bed sheets online through Shopify. Like every DTC (Direct-to-Consumer)
brand, they can look backwards easily: Shopify tells them how much they sold last month,
which products moved, and what the average order was. But they cannot look *forwards*.
They don't know which of their 751,956 customers will come back and which ones have
quietly stopped caring. They don't know when to reach out, what to offer, or who is
worth spending money to retain. RetentionIQ is the layer that answers those forward-looking
questions — and the notebooks are where we prove it's possible.

---

## NOTEBOOK 01 — Exploratory Data Analysis

### Section 1: Loading the Data — Why We Look Before We Model

The very first thing any serious data scientist does is load the data and simply *look* at
it. This is not optional ceremony — it is the most important step in any project. Before
you can build a model you need to know: How many rows do I have? What are the column names?
Are there nulls? What are the data types? Are the dates formatted correctly? Is there
anything obviously wrong?

In this notebook we load two files. The first, `orders_raw.csv`, has one row for every
order placed on the Miracle Sheets Shopify store — 976,659 rows in total. Each row tells
us which customer placed the order, when they placed it, how much they paid, and whether
it was paid, refunded, or voided. The second file, `customer_features.csv`, is a
pre-aggregated table with one row per customer — 751,956 rows — summarising each customer's
full purchase history into computed fields like total orders, total revenue, and average
days between purchases. This two-table structure (one row per event, one row per entity)
is the foundation of virtually every customer analytics project.

---

### Section 2: Schema Exploration — Knowing Your Columns

A schema is simply the structure of a dataset: the column names, their data types, and
what they mean. This matters enormously because computers are literal — if a date is stored
as text instead of a proper datetime type, calculations like "days since last order" will
fail or give wrong results. In this section we print the column names and data types for
both files, look at the null counts (missing values), and read a few sample rows.

The key finding is that the orders file has only 7 columns — customer ID, order ID, order
tags, order date, amount charged, product amount, and financial status. Notably absent is
a `discount_code` column, which matters later. The customer features file has 16 columns,
pre-computing things like frequency, recency, monetary value, and refund rate. Checking
for nulls tells us where data is incomplete, which is important because machine learning
models cannot accept missing values without special handling.

---

### Section 3: Orders Overview — Descriptive Statistics

Descriptive statistics are the foundation of any analysis. They answer the question "what
does this data look like in summary?" without trying to explain why or predict anything.
The key statistics here are:

- **Count**: 976,659 total orders over about two years (January 2024 to March 2026)
- **Revenue**: roughly $155.9 million in total charged across all valid orders
- **Average Order Value (AOV)**: $170.60 — this is the mean spend per order
- **Median Order Value**: $151.11 — this is the middle value when all orders are sorted

The difference between mean and median is important and worth understanding deeply. The
*mean* is calculated by adding everything up and dividing by the count. The *median* is
the value at the exact middle of the sorted list. When these two numbers differ a lot, it
tells you the distribution is *skewed* — there are some very large orders pulling the mean
upward, making the average look higher than what a typical customer actually spends. The
median is usually more honest for "typical customer behaviour" questions.

The monthly trend chart reveals seasonality — a huge spike in November 2025 (Black Friday
/ Cyber Monday), with the rest of the year more steady. This matters because any model
trained on this data will see November as unusual, and we need to be careful not to let
that period distort our understanding of normal behaviour.

The financial status breakdown shows that 89.3% of orders are "paid", 6.4% are fully
refunded, and 4.2% are partially refunded. For most analyses we exclude refunded and
voided orders, because they represent cancelled transactions that should not count as
real customer behaviour.

---

### Section 4: Customer Overview — The One-Time Buyer Problem

This is where the story really begins. We group the orders by customer and count how many
orders each person placed. The result is striking: **87.8% of Miracle Sheets' 751,956
customers have placed exactly one order**. Only 12.2% (about 92,000 people) have ever
come back for a second purchase.

This is not unusual for a DTC brand. Sheets are a durable good — you buy them, you use
them for years, and maybe you come back eventually. But "eventually" is hard to predict.
The histogram of orders per customer shows a massive bar at 1, then rapidly diminishing
bars at 2, 3, 4 — this is called a *heavy-tailed* or *right-skewed* distribution.

Understanding this single fact — 87.8% one-time buyers — completely defines the product
strategy. The entire value of RetentionIQ is identifying which of those one-time buyers
will convert to repeat customers if reached at the right moment, and which of the repeat
customers are starting to drift away. If we can move even 5% of one-time buyers to a
second purchase, the revenue impact is enormous.

---

### Section 5: Inter-Purchase Time — The Heartbeat of the Dataset

The inter-purchase time is the number of days between consecutive orders from the same
customer. It is the most important piece of timing information in the entire project,
and computing it requires a technique called a *window function* (specifically, the `lag`
operation): for each customer, we sort their orders by date, then look at the previous
order date and compute the gap in days.

We computed 118,006 such intervals — one for every time a repeat customer made a
subsequent purchase. The distribution of these intervals tells us the natural rhythm of
Miracle Sheets customers. The key statistics are:

- **Median (P50): 59 days** — half of all repeat purchases happen within 59 days of the
  previous one. This is surprisingly fast for a durable goods brand.
- **P75: 167 days** — three-quarters of repeat purchases happen within about 5.5 months
- **P90: 318 days** — 90% of repeat purchases happen within about 10.5 months
- **P95: 404 days** — 95% within about 13 months

The percentile notation (P50, P75, P90) is a way of describing where values fall in a
ranked list. P90 = 318 days means: if you take all 118,006 inter-purchase intervals and
sort them from smallest to largest, the value at the 90th position out of every 100 is
318 days. In other words, 90% of customers who will ever come back have already done so
within 318 days.

This number — **318 days** — becomes our `CHURN_WINDOW_DAYS` setting in Notebook 02. It
is not a guess. It is derived entirely from what real customers actually do.

---

### Section 6: Revenue Distribution — The Pareto Curve

The Pareto principle, often called the "80/20 rule", states that in many systems a small
proportion of inputs account for a large proportion of outputs. The classic formulation is
that 20% of customers generate 80% of revenue. This principle was observed by Italian
economist Vilfredo Pareto looking at land ownership in 19th-century Italy, and it turns
out to apply remarkably well to customer revenue distributions.

To visualise this, we draw a *Lorenz curve* (also called a Pareto curve in business
contexts). We sort all customers from highest to lowest spender, then plot cumulative
revenue (y-axis) against cumulative percentage of customers (x-axis). If every customer
spent exactly the same amount, the curve would be a perfect diagonal line (the dotted
line on the chart). The further the actual curve bows away from that diagonal, the more
concentrated the revenue is.

For Miracle Sheets, the result is: **the top 20% of customers generate 40% of revenue**.
This is actually less concentrated than the textbook 80/20 — the Pareto ratio here is
40%, not 80%. This is an interesting finding: revenue is more evenly spread than typical.
The implication is that RetentionIQ cannot focus only on protecting the top VIP customers
— broad retention across the whole base matters here.

The Average Order Value distribution separately shows the histogram of individual order
values. With a median of $151 and a mean of $170, we know the distribution is
right-skewed (pulled upward by expensive orders) but not dramatically so.

---

### Section 7: Refund Patterns

Refund behaviour is a proxy for customer dissatisfaction. A customer who has been refunded
once is statistically more likely to churn than one who hasn't. In our data, 7.1% of
customers have at least one refund, and the average refund rate per customer is also 7.1%.

We also note a data gap here: there is no `discount_code` column in `orders_raw.csv`.
Discount usage — whether a customer only buys when there's a promo code — is one of the
strongest predictors of long-term churn in e-commerce. Customers who exclusively buy with
discounts tend to have lower lifetime value because they won't pay full price. This field
needs to be added to the Shopify data extraction in the next sprint.

---

### Section 8: Churn Window Analysis — Turning a Business Question into a Number

"Churn" is a concept, not a measurement. In a subscription business (Netflix, Spotify),
churn is easy to define: did they cancel? In a non-contractual business like an online
shop, there is no cancellation event. A customer simply stops ordering. So we have to
*decide* when to call someone churned — and this decision must be grounded in data.

The sensitivity analysis in this section asks: if we declare a customer churned after
N days of silence, what percentage of real re-purchases would we have already captured?

| Window | % of repeat purchases captured |
|--------|-------------------------------|
| 60 days | 52.9% — misses too many slow buyers |
| 90 days | 61.8% |
| 180 days | 76.9% |
| **318 days** | **90.1% ← selected** |
| 365 days | 93.0% |

Choosing 90 days (the common industry default) would mean we label 38% of customers
as "churned" who are actually just slow repeat buyers — they would have come back, but
we'd have already written them off. 318 days captures 90% of real repurchases, meaning
only 10% of customers who eventually return are labelled as churned. This is the P90
threshold.

The survival curve plot visualises this beautifully: the x-axis is days since last order,
and the y-axis is the percentage of repeat customers who have repurchased by that day. The
curve rises steeply at first (fast repurchasers) then flattens out. The point where it
starts to flatten meaningfully — around 318 days — is where we draw our line.

---

## NOTEBOOK 02 — Model Training and Benchmarking

### Section 1: From Data to Features — The RFM Framework

Before fitting any model, we need to turn raw orders into a structured table of
*features* — one row per customer, with columns that summarise their behaviour as
numbers a model can learn from. The most important framework for this in customer analytics
is called **RFM**: Recency, Frequency, Monetary.

- **Recency** — how long ago was the customer's last purchase? A customer who bought
  yesterday is more likely to buy again than one who bought two years ago.
- **Frequency** — how many times has the customer bought? In the BG/NBD model specifically,
  this is defined as the number of *repeat* purchases (total orders minus 1), because the
  model needs to observe at least one repeat to estimate a purchase rate.
- **Monetary** — what is the average value of their orders? This feeds into the
  Gamma-Gamma model for predicting future spend.
- **T (tenure)** — how long has this customer been in the database? A customer who joined
  two years ago and bought 5 times has a very different purchase rate than one who joined
  two months ago and also bought 5 times.

These four numbers (F, R, T, M) are sufficient to fit the CLV model. For the churn model,
we compute additional features like `days_since_last_order`, `total_revenue`, and
`refund_rate`.

---

### Section 2: The BG/NBD Model — Predicting Future Purchase Behaviour

The **Beta-Geometric / Negative Binomial Distribution** model, introduced by Fader and
Hardie (2005), is the gold standard for predicting how many times a customer will purchase
in the future in a non-contractual setting. Understanding it requires understanding two
simpler ideas first.

The **Negative Binomial Distribution (NBD)** models the number of times an event (a
purchase) occurs in a fixed time period, given that different customers have different
rates of purchasing. Some customers buy frequently; some rarely. We model this variation
across customers with a *Gamma distribution* — a flexible mathematical shape that can
represent this kind of spread. Together, the Gamma prior on purchase rate and the Poisson
process for each individual customer produces the NBD. This is called a
*Gamma-Poisson mixture*.

The **Beta-Geometric (BG)** component models the fact that customers can become
permanently inactive — they "die" (in modelling terms). After each purchase, a customer
has some probability of never buying again. We model this probability as drawn from a
*Beta distribution*, which can flexibly represent populations where some people drop out
quickly and others stay for a long time. This is called *heterogeneous dropout*.

The BG/NBD model fits four parameters to the data:
- **r = 0.0353** and **alpha = 6.7611** control the distribution of purchase rates.
  r is very small, which means the Gamma distribution is highly skewed — most customers
  have very low purchase rates (consistent with 87.8% buying only once).
- **a = 0.5016** and **b = 0.2094** control the distribution of dropout probabilities.
  Since `a > b`, the model believes most customers are more likely to drop out than to
  stay — which again makes sense for a durable goods brand with infrequent repurchase.

Once fitted, the model can answer two questions for any customer: "How many purchases will
this customer make in the next N days?" and "What is the probability this customer is
still 'alive' (has not permanently dropped out)?"

The diagnostic heatmaps visualise this. The frequency-recency matrix shows that customers
who have bought many times and bought recently are expected to buy a lot more. The
probability-alive matrix shows that customers who have been dormant for a long time (and
have a high frequency of historical purchases) are likely to have dropped out — the model
correctly identifies this as a warning sign.

---

### Section 3: The Gamma-Gamma Model — Predicting Spend per Purchase

Knowing how many purchases a customer will make is not enough — we also need to know how
much they'll spend each time. The **Gamma-Gamma model** (Fader, Hardie & Lee, 2005) does
exactly this. It models the monetary value of individual transactions under two assumptions:
(1) the spend of each transaction is independent of the number of transactions (frequency
and spend are uncorrelated), and (2) each customer has their own average spend level,
which itself follows a Gamma distribution.

The three fitted parameters are:
- **p = 4.0817** and **q = 0.4348** and **v = 3.8700** — these define the shape of the
  Gamma distribution over average spend. The specific values here reflect that Miracle
  Sheets customers' spend is somewhat concentrated (sheets come in a limited price range)
  but with meaningful variation across customers.

---

### Section 4: Customer Lifetime Value — Combining the Two Models

**Customer Lifetime Value (CLV)** is the total net revenue a business expects from a
customer over their remaining relationship. It is the single most strategically important
number in customer analytics, because it answers the question: "How much is this customer
worth to invest in?"

By combining the BG/NBD model (how many purchases?) with the Gamma-Gamma model (how much
per purchase?), we compute a predicted 12-month CLV for every customer. The formula is
essentially: expected number of future purchases × expected spend per purchase, discounted
for the time value of money (we use a 1% monthly discount rate, approximately 12% annual).

The results for the 92,021 repeat customers (the only ones with a non-zero monetary value
to feed the Gamma-Gamma model) are:

- **Median 12-month CLV: $14.72** — a typical repeat customer is worth about $15 over the
  next year.
- **Mean 12-month CLV: $48.98** — the mean is more than 3× the median, meaning a small
  group of very high-value customers is pulling the average upward dramatically.
- **P90 CLV: $118.31** — the top 10% of customers are worth more than $118 each.
- **Maximum: $18,576** — a single customer at the very top tail.

This right-skewed distribution (mean >> median) is typical of CLV distributions and
reinforces the Pareto finding: a small number of customers carry disproportionate future
value.

---

### Section 5: Temporal Split — The Most Important Methodological Decision

Before training any churn model, we face a critical methodological question: how do we
split our data into training and test sets? In most machine learning tutorials, the
standard answer is "randomly" — shuffle the data and put 75% in training, 25% in testing.
For customer churn prediction, **this is wrong and will give you dangerously misleading results**.

Here is why. Our features are things like "total orders", "days since last order", and
"total revenue". Our label is "did this customer churn?" If we split randomly, a customer's
test record might use features computed from their full order history — including orders
that happened *after* the date we're supposedly predicting from. The model would be
learning from the future to predict the future. In production, that future data won't
exist. The model's performance in the real world will be far worse than on your test set.
This is called **data leakage**.

The correct method is a **temporal split**:
1. Choose a *split date* = maximum order date − churn window (2026-03-14 − 318 days = 2025-04-30)
2. Compute all *features* using only orders that happened **before** the split date
3. Compute *labels* based on whether the customer ordered **after** the split date
4. Train on a random subset of customers, test on the rest — but both use the same
   temporal boundary

This gives us 384,699 customers in our churn dataset. Of those, 93.9% (361,222) are
labelled "churned" (no order in the 318-day window after the split date) and only 6.1%
(23,477) are "active". This severe imbalance matters enormously and is handled separately.

---

### Section 6: Class Imbalance — When 93.9% is One Answer

**Class imbalance** is when one outcome is much more common than the other in a
classification problem. Here, 93.9% of customers are churned. If a model simply predicted
"churned" for every single customer without learning anything at all, it would be correct
93.9% of the time. That sounds good on the surface but is completely useless — it would
never identify the active customers we actually want to retain.

We handle this in two ways. For Logistic Regression and Random Forest, we use the
`class_weight='balanced'` parameter, which automatically re-weights each class so the
model penalises getting the minority class (active customers) wrong much more heavily.
For XGBoost, we use the `scale_pos_weight` parameter, which is set to the ratio of
negative to positive samples — approximately 0.065 (6,100 active customers / 361,222
churned customers ≈ 0.065). This tells XGBoost to treat each active customer as if they
were worth about 15 churned customers in the loss function.

---

### Section 7: Logistic Regression — The Simplest Possible Model

**Logistic Regression** is the simplest classification model and is used as a *baseline*.
A baseline is intentionally simple — its purpose is to set a floor. If even a basic model
can solve the problem, you don't need complexity. If a complex model only marginally
outperforms the baseline, you should question whether the complexity is worth it.

Logistic Regression works by learning a *linear combination* of features that predicts a
probability between 0 and 1. Each feature gets a weight, those weights are multiplied by
the feature values and summed, and then the result is squashed through a sigmoid function
to produce a probability. The model finds the weights that best separate churned from
active customers.

Its limitation is the word "linear" — it can only draw a straight-line boundary between
classes in the feature space. Real customer behaviour is not neatly separable by a line.
Our Logistic Regression achieved **AUC-ROC 0.6718**.

---

### Section 8: The AUC-ROC Metric — How We Measure Churn Model Quality

The **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve) is the standard
metric for binary classification problems, and it deserves a careful explanation.

First, the ROC curve. For every possible *threshold* (a cut-off between 0 and 1 above
which we declare a customer "churned"), we compute two numbers:
- **True Positive Rate (Recall/Sensitivity)**: of all the actually-churned customers,
  what fraction did the model correctly flag? Higher is better.
- **False Positive Rate**: of all the actually-active customers, what fraction did the
  model wrongly flag as churned? Lower is better.

By sweeping the threshold from 0 to 1, we trace a curve — the ROC curve. The AUC is the
area under that curve. It ranges from 0.5 (random guessing, diagonal line) to 1.0 (perfect
model). An AUC of 0.68 means: "if I pick one randomly-chosen churned customer and one
randomly-chosen active customer and show them to the model, the model will correctly
rank the churned one as higher-risk 68% of the time."

Our AUC values of 0.67–0.68 are modest. This is expected given two constraints: the
extreme class imbalance (93.9% churned), and the limited feature set (only 9 RFM-type
features, no discount usage, no product diversity). The target in the guidelines is
AUC > 0.75, which is achievable once discount codes and product affinity features are
added in the next ingestion sprint.

---

### Section 9: Random Forest — Ensembles of Decision Trees

A **Random Forest** is an *ensemble* method — it builds many individual decision trees,
each trained on a slightly different random subset of the data and a random subset of
features, and then averages their predictions. The key insight is that while a single
decision tree can be very sensitive to noise in the training data (a small change in the
data gives you a completely different tree), averaging hundreds of trees smooths out that
noise. This is called *variance reduction*.

Each individual decision tree learns a series of yes/no questions about the features:
"Is days_since_last_order > 200? If yes, is total_orders == 1? If yes → likely churned."
A Random Forest makes each tree ask slightly different questions and then votes. The
majority answer wins.

Random Forests handle non-linear relationships naturally (unlike Logistic Regression) and
are robust to outliers. They don't need feature scaling. Our Random Forest achieved
**AUC-ROC 0.6814** — a modest improvement over Logistic Regression, confirming there is
non-linear structure in the data but that our feature set is the limiting factor.

---

### Section 10: XGBoost — Gradient Boosting, the State of the Art

**XGBoost** (eXtreme Gradient Boosting, Chen & Guestrin, 2016) is consistently the
top-performing algorithm on structured / tabular data in machine learning competitions and
industry applications. It is the algorithm of choice for churn prediction in the academic
literature (De Caigny et al., 2018).

Where Random Forest builds trees *in parallel* (independently), XGBoost builds them
*sequentially* through a process called **gradient boosting**. Each new tree focuses on
correcting the mistakes made by all the previous trees combined. The "gradient" refers to
the mathematical gradient of the loss function — each new tree learns to push predictions
in the direction that reduces the error most. The "boosting" refers to the sequential,
error-correcting nature of the process.

XGBoost has several hyperparameters that were set explicitly:
- `n_estimators=300` — build up to 300 trees
- `max_depth=6` — each tree can ask at most 6 yes/no questions deep
- `learning_rate=0.05` — each new tree only makes small corrections (preventing overfitting)
- `early_stopping_rounds=30` — stop if the test AUC hasn't improved in 30 trees
- `scale_pos_weight≈15` — upweight minority (active) class

XGBoost achieved **AUC-ROC 0.6819** — the best of the three models, and selected as the
production model for RetentionIQ.

---

### Section 11: The Precision-Recall Curve — A Second View of Model Quality

Alongside the ROC curve, we also plot the **Precision-Recall curve**. This is particularly
informative when classes are highly imbalanced, like ours.

- **Precision**: of all customers the model flags as churned, what fraction are actually
  churned? A model with low precision sounds many false alarms.
- **Recall**: of all customers who will actually churn, what fraction does the model
  catch? A model with low recall misses too many at-risk customers.

There is always a trade-off between precision and recall: if you make the threshold very
low (flag everyone as churned), recall is 1.0 but precision is terrible. The
**Average Precision (AP)** score summarises this trade-off into one number. Our XGBoost AP
is 0.9670, which looks very high — but with 93.9% churn rate, even a naive model can
achieve high AP by just predicting "churned" constantly. The ROC AUC is a more honest
metric in this highly-imbalanced setting.

---

### Section 12: SHAP Values — Making the Black Box Explain Itself

**SHAP** (SHapley Additive exPlanations, Lundberg & Lee, 2017) is a method for explaining
individual model predictions. It is named after Lloyd Shapley, who invented the underlying
game-theory concept in the 1950s (and won the Nobel Prize in Economics in 2012 for related
work).

The core idea comes from cooperative game theory. Imagine each feature as a "player" in a
game, where the "game" is producing a prediction. SHAP asks: "how much did each player
contribute to the final score?" It does this by considering every possible combination of
features being present or absent, and measuring how much the prediction changes when each
feature joins the coalition. The SHAP value for a feature is its average marginal
contribution across all possible coalitions.

In practice, for tree-based models like XGBoost, this can be computed efficiently without
actually trying every combination. The result for each customer is a set of SHAP values —
one per feature — that sum to the difference between the customer's predicted churn
probability and the average churn probability across all customers.

The **summary plot** (plot 5 in `docs/figures/`) shows global feature importance: each
dot is a customer, the x-axis shows how much that feature pushed the prediction towards
or away from churn, and the colour shows whether the feature value was high (red) or low
(blue). Features at the top are most important overall.

The **waterfall plot** (plot 6) shows a single customer's explanation: starting from the
baseline prediction (average across all customers), each bar shows how much each feature
adjusted the prediction up or down to reach the final churn probability. This is the
"why is THIS customer at risk?" view — and it is RetentionIQ's core differentiator. No
other Shopify analytics tool currently offers this.

---
---

## Part 2: The Story for the Master's Director

---

# Director Presentation Story — RetentionIQ

## The Narrative Arc (How to Tell It in the Room)

Use this as your speaking guide — paragraph by paragraph, mapped to each slide.

---

### Opening: Start with the Problem, Not the Technology

*Slide 1 — Title*

> "The brand we're working with — Miracle Sheets — has excellent descriptive analytics.
> They know what happened last month, last quarter, last year. But the question their
> marketing team asks every Monday morning is different: which of our customers are about
> to disappear, and what can we do about it? That question is not answered anywhere in
> their existing stack. RetentionIQ is our answer."

This framing is important because it grounds the technical work in a real problem.
Directors want to see that the team understands the business, not just the algorithms.

---

### Slide 2: The Problem and the Dataset

> "We are working with real data — 976,659 orders from 751,956 customers, spanning
> January 2024 to March 2026. This is not synthetic data or a toy dataset. These are
> real Miracle Sheets customers, and the patterns we found are the actual patterns of
> the brand."

> "The first thing we discovered is that 87.8% of customers have bought exactly once.
> That single number defines the entire product strategy. These are not churned customers —
> they might come back. The question is: which ones, when, and what should we say to them?"

---

### Slide 3: Architecture — Connect Code to Stack

> "The architecture has five layers. At the bottom, Shopify data is stored in a cloud
> warehouse. In the middle, Python feature engineering turns raw orders into a table of
> 9 customer-level signals per customer. Those signals feed two models: one that predicts
> how much each customer is worth over the next 12 months, and one that scores their
> probability of churning. At the top, a FastAPI backend serves those predictions to
> a Streamlit dashboard — or back into the existing Looker reporting environment the
> brand already uses."

> "Every step is logged, versioned, and reproducible. Every prediction carries the date
> it was made and the model version that made it."

---

### Slide 4: Model Choice — Lead with the Empirical Churn Window

This is where the EDA work pays off in the presentation. Don't just say "we used XGBoost."
Show that the entire model design was driven by data, starting with the churn window.

> "Before we could label a single customer as churned, we had to answer a fundamental
> question: what does 'churned' actually mean for a brand that sells bed sheets? A sheet
> lasts years. The 90-day churn window you see in textbooks and SaaS tools was designed
> for subscription businesses. We measured the actual inter-purchase time distribution
> for Miracle Sheets customers and found that the 90th percentile is 318 days — roughly
> 10 months. Using 90 days would have misclassified 38% of slow-but-loyal customers as
> churned. Our churn window is 318 days, and it is fully empirical."

> "With that window defined, we implemented a strict temporal split. Features are
> computed from orders before April 30 2025. Labels are set by whether the customer
> ordered after that date. This eliminates data leakage — a common methodological error
> that inflates test metrics in academic submissions. We benchmarked three models on
> 384,699 customers."

> "Logistic Regression — our baseline — scored AUC-ROC 0.6718. This is the floor: how
> well can a simple linear model do? Random Forest improved marginally to 0.6814, showing
> there is non-linear structure the linear model misses. XGBoost reached 0.6819 and is
> our production model."

> "The AUC values are modest, and we want to be transparent about why. With 93.9% churn
> rate and only 9 RFM features — because our current dataset lacks discount codes and
> product-level detail — we are near the ceiling of what these features can predict.
> The literature (De Caigny et al., 2018) confirms that AUC typically jumps 5-10 points
> when behavioural features like discount dependency and product diversity are added.
> Adding those features is the first priority of the next sprint."

---

### Slide 5: Literature — Connect to the Academic Foundations

> "Our model choices are not arbitrary. The BG/NBD model was introduced by Fader and
> Hardie in 2005 in Marketing Science and remains the gold standard for non-contractual
> CLV. It is used by Shopify's own analytics team, by Starbucks, and by Spotify. The
> Gamma-Gamma extension for monetary value is from the same authors, the same year."

> "SHAP is from Lundberg and Lee's 2017 NeurIPS paper, which has over 15,000 citations.
> It is the most principled approach to model explanation currently available, grounded
> in Shapley values from cooperative game theory — work that earned Lloyd Shapley the
> 2012 Nobel Prize in Economics."

> "The churn benchmarking methodology follows De Caigny et al. (2018), who showed in a
> large-scale empirical study that gradient boosting consistently outperforms linear models
> for customer churn, which our results confirm."

---

### Slide 7: Demo — What You Will Show Live

Walk through this in the following order for maximum impact:

1. **Open Notebook 01, Section 8** — show the survival curve and explain the 318-day
   churn window in your own words. This is your empirical methodology moment.

2. **Open Notebook 02, the ROC curve** — show the three models on one chart. Explain
   why the curves are close together and what that tells you about the feature set.

3. **Open the SHAP waterfall plot** — zoom in and read it out loud for a single customer:
   *"This customer has a 0.92 probability of churning. The biggest factor pushing that
   number up is their days_since_last_order of 287 days — shown by the red bar on the
   right. The only factor working in their favour is a high average order value — the
   blue bar pushing left. The model is saying: this person used to spend well, but they
   haven't been seen in almost 10 months."*

   This moment — explaining a single customer's churn risk in plain English from a
   model output — is your strongest demonstration. No other Shopify tool does this.

---

### Handling Hard Questions

**"Why is your AUC only 0.68? That seems low."**

> "It is lower than our target of 0.75, and we are being transparent about why. We have
> a 93.9% churn rate, which means the class imbalance is severe. We also have only 9
> features, because our current data extract does not include discount codes or product
> detail. The literature shows these features alone typically add 5-10 AUC points. The
> model architecture is correct — the data is the constraint, and it is already in the
> next sprint backlog."

**"Why not use a neural network?"**

> "For tabular data at this scale, gradient boosting consistently outperforms neural
> networks in empirical benchmarks (Grinsztajn et al., 2022). Neural networks require
> much larger datasets to converge and are significantly harder to explain to a business
> user. XGBoost with SHAP gives us better performance, full explainability, and 100x
> faster inference — which matters for a real-time API endpoint."

**"How do you validate the CLV predictions?"**

> "BG/NBD CLV models are validated by holding out a calibration period and a holdout
> period. You fit the model on, say, the first 18 months, predict CLV for months 19-24,
> and compare to actual revenue in months 19-24. With two years of data we have enough
> history to run this validation. It is the next step in the model evaluation notebook
> (planned for Notebook 06)."

**"Is the data privacy-compliant?"**

> "Customer emails are hashed with SHA-256 before entering the ML pipeline. No PII
> appears in model features, predictions, or the dashboard. The Snowflake warehouse
> is SOC 2 Type II certified, with role-based access control giving the ML service
> account read-only access. All credentials are stored in `.env` files loaded by
> `python-dotenv` and are excluded from the Git repository."

---

### Closing: End on the Business Impact

> "The numbers we showed today — 751,956 customers, 87.8% buying only once, top 20%
> generating 40% of revenue — define the size of the problem. A 5-point improvement
> in repeat purchase rate across this base is worth millions in incremental revenue.
> RetentionIQ gives the brand operator the information they need to target the right
> customer, with the right message, at the right time. And crucially — it tells them
> *why*. That explainability is what separates a tool that gets used from one that gets
> shelved."

---

*End of director_explanation.md*
