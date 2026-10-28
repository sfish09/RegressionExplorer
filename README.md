This is an interactive Streamlit app for exploring data and building regression models, 
with no coding needed. Upload any CSV, choose a response and predictors, and the app 
walks through plots, model fitting, hypothesis testing, and cross-validation. 

Files:
app.py is the entire application
requirements.txt lists the packages the cloud server must install
READE.md is this file

Design Notes:
- Streamlit reruns the whole script on every interaction. Results that sruvive the rerun are stored in st.session_state
- st.set_page_config must be the first Streamlit call in the script
- st.stop() after the uploader prevents the rest of the script from running before a file exists
- PolynomialFeatures and OneHotEncoder are fit ont he training folds only

What app does:
1. Uploead and clean - upload a csv, declare missing-value markers, auto-convert columns, exclude columns such as IDs or names
2. Data overview - dimensions, preview, column types, summary statistics, missing value counts
3. Variable selection - pick a numeric reponse, choose predictors, and decide which are categorical
4. Explore - histogram, boxplot, and scatterplot with color picker, bin slider and optional trend line
5. Linear regression - fit a multiple regression on numeric predictors, shows R-squared, RSS, RSE and coefficients
6. Polynomial experiment - fit a polynomial of degree 1 to 5 fo rone variable and plot the curve
7. Residual diagnostics - residuals versus fitted values or any predictor, plus a residual histogram, for the linear and polynomial model
8. Categorical encoding - preview of dumy vairables and the reference category
9. Interaction experiment - compare an additive model with one that indcludes a product term
10. Coefficient t-test - standard errors, t-statistics, p-values, decisions, and  a plot of the t distribution with the rejection regions
11. Nested F-test - compare reduces and full models on identical complete-case rows; categorical variables enter as all k-1 dummies together
12. 10-fold CV - compare polynomial degrees using the same folds every model
