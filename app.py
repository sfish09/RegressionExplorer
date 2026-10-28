# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:09:04 2026

@author: sthfi
"""

# Part 1
# Import the packages we will use
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Page settings (must come before other st commands)
st.set_page_config(page_title="Regression Explorer", layout="centered")

st.title("Regression Explorer")
st.write("Upload a CSV to begin your analysis.")

# Part 2
#Add a file picker button
file = st.file_uploader("Choose a CSV file", type="csv")
if file is None:
    st.info("Upload a dataset.")
    st.stop()
#reads csv file from the uploaded file
#data = pd.read_csv(file)
# Part 13: Data cleaning controls
# Let the user declare extra symbols that mean "missing" (e.g., ? or -999)
missing_text = st.text_input(
    "Extra missing-value markers (comma-separated)", value="?")
na_list = [s.strip() for s in missing_text.split(",") if s.strip()]

# Read the CSV, treating those markers as missing (NaN)
data = pd.read_csv(file, na_values=na_list, skipinitialspace=True)

# Convert text columns that are really numbers (mostly numeric entries)
if st.checkbox("Convert text columns that look numeric", value=True):
    for c in data.columns:
        if not pd.api.types.is_numeric_dtype(data[c]):
            converted = pd.to_numeric(data[c], errors="coerce")
            n_filled = data[c].notna().sum()
            if n_filled > 0 and converted.notna().sum() >= 0.9 * n_filled:
                data[c] = converted

# Let the user drop columns that should not be analyzed (IDs, names, etc.)
drop_cols = st.multiselect("Columns to exclude from the analysis",
                           list(data.columns))
data = data.drop(columns=drop_cols)
if data.shape[1] == 0:
    st.error("All columns were excluded.")
    st.stop()
#Add subhead
st.subheader("Data overview")
#Display text
st.write("Dimensions (rows, columns):", data.shape)
#display the first 10 rows in a streamlit dataframe
st.dataframe(data.head(10))
# Editable table: replaces st.dataframe(data.head(10))
#st.write("Edit the data below (double-click a cell, or add/delete rows):")
#data = st.data_editor(data, num_rows="dynamic", key="data_editor")
st.write("Column types:")
st.dataframe(data.dtypes.astype(str))
st.write("Summary statistics:")
st.dataframe(data.describe(include="all").T)
st.write("Missing values:")
st.dataframe(data.isna().sum())

# Part 3
# Find all columns that hold numbers (int or float)
# Only these can serve as the response (Y) in the regression
numeric = data.select_dtypes(include=np.number).columns.tolist()
# If the dataset has no numeric columns, show an error and stop the app
# since there is nothing to predict
if not numeric:
    st.error("No numeric response variable found.")
    st.stop()
# Dropdown to pick ONE response variable from the numeric columns
response = st.selectbox("Response variable", numeric)
# Predictor candidates: every columns except one chosen as the response 
candidates = [c for c in data.columns if c != response]
# Multi-select box to pick one or more predictor variables (X)
predictors = st.multiselect("Predictor variables", candidates)
# Text column can't be used as numbers, so pre-selected them as categorical
# This keeps every chosen predictor that is not in the numberic list
default_cats = [c for c in predictors if c not in numeric]
# Let the user choose which predictor to treat as categorical
# Options are limited to the chosen predictors
categorical = st.multiselect(
    "Treat selected predictors as categorical",
    predictors, default=default_cats
)
# Warn about categorical variables with too many levels
for c in categorical:
    k = data[c].nunique()
    if k > 15:
        st.warning(f"'{c}' has {k} levels, which creates {k-1} dummy "
                   "columns. Consider excluding it or choosing a different "
                   "variable.")
# Whatever is not categorical is quantitative 
quantitative = [c for c in predictors if c not in categorical]
# Display the final split so the user can confirm their choices
st.write("Quantitative:", quantitative)
st.write("Categorical:", categorical)

# Part 4
st.subheader("Explore the data")
plot_type = st.radio("Plot type", ["Histogram", "Boxplot", "Scatterplot"])
plot_col = st.selectbox("Numeric variable", numeric, key="plot_col")

# Customization 1: let the user pick the plot color
color = st.color_picker("Plot color", "#1f77b4")

fig, ax = plt.subplots()
if plot_type == "Histogram":
    # Customization 2: slider controls the number of bins
    bins = st.slider("Number of bins", 5, 50, 20)
    ax.hist(data[plot_col].dropna(), bins=bins, color=color, edgecolor="black")
elif plot_type == "Boxplot":
    ax.boxplot(data[plot_col].dropna(), patch_artist=True,
               boxprops=dict(facecolor=color))
else:
    ax.scatter(data[plot_col], data[response],color=color, alpha=0.5)
    ax.set_ylabel(response)
    # Customization 3: optional trend line
    if st.checkbox("Show trend line"):
        pair = data[[plot_col, response]].dropna()
        slope, intercept = np.polyfit(pair[plot_col], pair[response], 1)
        xs = np.linspace(pair[plot_col].min(), pair[plot_col].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="red")
ax.set_xlabel(plot_col)
ax.set_title(f"{plot_type}: {plot_col}")
st.pyplot(fig)
plt.close(fig)

# Part 5
st.subheader("Linear regression")
fit_vars = st.multiselect("Numeric model predictors", quantitative)
if st.button("Fit regression"):
    if not fit_vars:
        st.warning("Select at least one numeric predictor.")
    else:
        clean = data[[response] + fit_vars].dropna()
        X = clean[fit_vars]
        y = clean[response]
        model = LinearRegression().fit(X, y)
        predicted = model.predict(X)
        residuals = y.to_numpy() - predicted
        n, p = X.shape
        rss = np.sum(residuals**2)
        if n <= p + 1:
            st.error("Insufficient residual degrees of freedom.")
        else:
            st.session_state["fit"] = dict(
                model=model, X=X, y=y, predicted=predicted,
                residuals=residuals, rss=rss,
                rse=np.sqrt(rss/(n-p-1)),
                r2=r2_score(y, predicted)
            )
if "fit" in st.session_state:
    r = st.session_state["fit"]
    st.write("R-squared:", round(r["r2"], 4))
    st.write("RSS:", round(r["rss"], 2))
    st.write("RSE:", round(r["rse"], 2))
    st.write("Intercept:", r["model"].intercept_)
    st.dataframe(pd.DataFrame({
        "Predictor": r["X"].columns,
        "Coefficient": r["model"].coef_
    }))

# Part 6
st.subheader("Residual diagnostics")
if "fit" in st.session_state:
    r = st.session_state["fit"]
    c1, c2 = st.columns(2)
    fig1, ax1 = plt.subplots()
    ax1.scatter(r["predicted"], r["residuals"], alpha=0.5)
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set(xlabel="Fitted", ylabel="Residual",
            title="Residuals versus fitted")
    c1.pyplot(fig1)
    plt.close(fig1)
    fig2, ax2 = plt.subplots()
    ax2.hist(r["residuals"], bins=20, edgecolor="black")
    ax2.set(title="Residual distribution", xlabel="Residual")
    c2.pyplot(fig2)
    plt.close(fig2)

# Part 7
st.subheader("Polynomial experiment")

if quantitative:
    poly_var = st.selectbox("Polynomial variable", quantitative)
    degree = st.slider("Degree", 1, 5, 2)
    sub = data[[response, poly_var]].dropna()
    X0 = sub[[poly_var]]
    y0 = sub[response]
    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = transformer.fit_transform(X0)
    poly_model = LinearRegression().fit(Xp, y0)
    # Display the polynomial regression coefficients
    st.write("Intercept:", poly_model.intercept_)

    coefficient_table = pd.DataFrame({
        "Term": transformer.get_feature_names_out(),
        "Coefficient": poly_model.coef_
        })

    st.dataframe(coefficient_table)
    st.write("Training R-squared:", round(poly_model.score(Xp, y0), 4))
    st.write("Terms:", transformer.get_feature_names_out())
    grid = pd.DataFrame({
        poly_var: np.linspace(X0[poly_var].min(),
                         X0[poly_var].max(), 200)
        })
    fig, ax = plt.subplots()
    ax.scatter(X0[poly_var], y0, alpha=0.3)
    ax.plot(grid[poly_var], poly_model.predict(
        transformer.transform(grid)), color="black")
    ax.set(xlabel=poly_var, ylabel=response, title=f"Degree {degree}")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Select a quantitative predictor above to try a polynomial fit")

# Checkpoint 7
st.subheader("Residual diagnostics")
# Build a lost of models that are available to inspect
model_options = []
if "fit" in st.session_state:
    model_options.append("Linear regression")
if quantitative:
    model_options.append("Polynomial regression")
if not model_options:
    st.info("Fit linear regression or choose a polynomial variable first.")
else:
    model_choice = st.radio("Model", model_options, horizontal=True)
    
    if model_choice == "Linear regression":
        r = st.session_state["fit"]
        fitted = r["predicted"]
        resid = r["residuals"]
        X_used = r["X"]
    else:
        fitted = poly_model.predict(Xp)
        resid = y0.to_numpy() - fitted
        X_used = X0
    # User picks what goes on the x-axis of the residual plot
    x_choice = st.selectbox("Plot residuals against", ["Fitted values"] + list(X_used.columns))
    if x_choice =="Fitted values":
        x_vals = fitted
    else:
        x_vals = X_used[x_choice].to_numpy()
    c1, c2 = st.columns(2)
    
    # Left: residuals versus the chosen x variable
    fig1, ax1 = plt.subplots()
    ax1.scatter(x_vals, resid, alpha=0.5)
    ax1.axhline(0, color="black", linestyle="--")
    ax1.set(xlabel=x_choice, ylabel="Residual",
            title=f"{model_choice}: residuals vs {x_choice}")
    c1.pyplot(fig1)
    plt.close(fig1)

    # Right: distribution of the residuals
    fig2, ax2 = plt.subplots()
    ax2.hist(resid, bins=20, edgecolor="black")
    ax2.set(title=f"{model_choice}: residual distribution",
            xlabel="Residual")
    c2.pyplot(fig2)
    plt.close(fig2)    
        
# Part 8
st.subheader("Categorical encoding")
if categorical and st.checkbox("Show dummy-variable preview"):
    raw = data[categorical].dropna().astype(str)
    enc = OneHotEncoder(drop="first", sparse_output=False)
    values = enc.fit_transform(raw)
    names = enc.get_feature_names_out(categorical)
    st.dataframe(pd.DataFrame(values, columns=names).head())
    for name, levels in zip(categorical, enc.categories_):
        st.write(f"{name}: reference category = {levels[0]}")
    
# Part 9
st.subheader("Interaction experiment")
if len(quantitative) >= 2:
    x1 = st.selectbox("First variable", quantitative, key="x1")
    x2 = st.selectbox("Second variable",
                      [x for x in quantitative if x != x1], key="x2")
    if st.button("Compare interaction"):
        sub = data[[response, x1, x2]].dropna().copy()
        sub["product"] = sub[x1] * sub[x2]
        y = sub[response]
        reduced = sub[[x1, x2]]
        full = sub[[x1, x2, "product"]]
        m0 = LinearRegression().fit(reduced, y)
        m1 = LinearRegression().fit(full, y)
        st.write("Additive R-squared:", m0.score(reduced, y))
        st.write("Interaction R-squared:", m1.score(full, y))
        st.dataframe(pd.DataFrame({
            "Term": full.columns, "Coefficient": m1.coef_
        }))

# Part 10
#hypothesis testing
st.subheader("Part 10: Coefficient Hypothesis Testing")
if "fit" not in st.session_state:
    st.info("Fit a quantitative regression model first.")
else:
    # Retrieve the last fitted regression model
    result = st.session_state["fit"]
    X = result["X"]
    y = result["y"]
    model = result["model"]
    # Let the user select a significance level
    alpha = st.selectbox(
        "Significance level (alpha)",
        [0.01, 0.05, 0.10],
        index=1
    )
    # Run the tests only when the button is pressed
    if st.button("Test coefficients"):
        # Number of observations and predictors
        n, p = X.shape
        # Residual degrees of freedom
        df = n - p - 1
        # Construct design matrix including intercept
        X_design = np.column_stack([
            np.ones(n),
            X.to_numpy(dtype=float)
        ])

        # Check degrees of freedom and matrix rank
        if n <= p + 1:
            st.error(
                "Insufficient observations for coefficient tests."
            )
        elif np.linalg.matrix_rank(X_design) < p + 1:
            st.error(
                "The design matrix does not have full column rank. "
                "Check for redundant or perfectly correlated predictors."
            )
        elif not np.isfinite(X_design).all():
            st.error("The predictor matrix contains invalid values.")
        else:
            # STEP 1: Obtain residuals and RSS
            predictions = model.predict(X)
            residuals = y.to_numpy() - predictions
            rss = np.sum(residuals ** 2)
            # STEP 2: Estimate error variance
            s_squared = rss / df

            # STEP 3: Coefficient covariance matrix
            XtX = X_design.T @ X_design
            XtX_inverse = np.linalg.inv(XtX)
            covariance_matrix = s_squared * XtX_inverse
            # STEP 4: Coefficient standard errors
            standard_errors = np.sqrt(
                np.diag(covariance_matrix)
            )
            # STEP 5: Estimated coefficients
            coefficients = np.concatenate([
                [model.intercept_],
                model.coef_.ravel()
            ])
            # STEP 6: Calculate t-statistics
            t_values = coefficients / standard_errors

            # STEP 7: Two-sided p-values
            p_values = 2 * stats.t.sf(
                np.abs(t_values),
                df=df
            )

            # STEP 8: Two-sided critical value
            t_critical = stats.t.ppf(
                1 - alpha / 2,
                df=df
            )
            # Build the coefficient table
            term_names = ["Intercept"] + list(X.columns)
            coefficient_table = pd.DataFrame({
                "Term": term_names,
                "Coefficient": coefficients,
                "Standard Error": standard_errors,
                "t-statistic": t_values,
                "p-value": p_values,
                "Decision": [
                    "Reject H0" if pv < alpha
                    else "Fail to reject H0"
                    for pv in p_values
                ]
            })
            # Save results so the plot can update independently
            st.session_state["t_test_results"] = {
                "table": coefficient_table,
                "df": df,
                "alpha": alpha,
                "t_critical": t_critical,
                "n": n,
                "p": p,
                "s_squared": s_squared
            }
    # DISPLAY THE RESULTS
    if "t_test_results" in st.session_state:
        test = st.session_state["t_test_results"]
        # Only show results for the alpha value used
        if test["alpha"] == alpha:
            st.write("Number of observations:", test["n"])
            st.write("Number of predictors:", test["p"])
            st.write("Residual degrees of freedom:", test["df"])
            st.write(
                "Estimated error variance:",
                round(test["s_squared"], 4)
            )
            st.write(
                "Critical t-values:",
                round(-test["t_critical"], 4),
                round(test["t_critical"], 4)
            )
            st.dataframe(
                test["table"].style.format({
                    "Coefficient": "{:.4f}",
                    "Standard Error": "{:.4f}",
                    "t-statistic": "{:.4f}",
                    "p-value": "{:.4g}"
                })
            )
            # VISUALIZE THE T-TEST
            st.subheader("Visualize an Individual Coefficient Test")
            selected_term = st.selectbox(
                "Select a coefficient",
                test["table"]["Term"].tolist()
            )
            selected_row = test["table"].loc[
                test["table"]["Term"] == selected_term
            ].iloc[0]

            observed_t = float(selected_row["t-statistic"])
            observed_p = float(selected_row["p-value"])

            df = test["df"]
            critical = test["t_critical"]
            # Set horizontal range for the graph
            limit = max(
                4,
                abs(observed_t) + 1,
                critical + 1
            )
            x_values = np.linspace(-limit, limit, 1000)
            y_values = stats.t.pdf(x_values, df=df)
            fig, ax = plt.subplots(figsize=(9, 5))
            # Plot Student's t-distribution
            ax.plot(
                x_values,
                y_values,
                color="navy",
                label=f"t-distribution (df={df})"
            )

            # Shade left rejection region
            ax.fill_between(
                x_values,
                y_values,
                where=x_values <= -critical,
                color="red",
                alpha=0.25
            )
            # Shade right rejection region
            ax.fill_between(
                x_values,
                y_values,
                where=x_values >= critical,
                color="red",
                alpha=0.25,
                label="Rejection regions"
            )
            # Critical limits
            ax.axvline(
                -critical,
                color="red",
                linestyle="--",
                label=f"Critical values: ±{critical:.3f}"
            )
            ax.axvline(
                critical,
                color="red",
                linestyle="--"
            )
            # Observed t-statistic
            ax.axvline(
                observed_t,
                color="green",
                linewidth=2.5,
                label=f"Observed t = {observed_t:.3f}"
            )
            ax.set_xlabel("t-statistic")
            ax.set_ylabel("Probability density")
            ax.set_title(
                f"Two-sided t-test: {selected_term}"
            )
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)
            # STATISTICAL CONCLUSION
            st.write("H0: The population coefficient equals zero.")
            st.write("H1: The population coefficient is not zero.")

            st.write("Observed t:", round(observed_t, 4))
            st.write("Two-sided p-value:", round(observed_p, 6))

            if observed_p < alpha:
                st.success(
                    f"Reject H0 at alpha = {alpha}. "
                    "The coefficient is statistically different "
                    "from zero under the model assumptions."
                )
            else:
                st.info(
                    f"Fail to reject H0 at alpha = {alpha}. "
                    "There is insufficient evidence that the "
                    "coefficient differs from zero."
                )

# Part 11: Nested F-test
st.subheader("Part 11: Nested F-test")

if len(predictors) < 1:
    st.info("Select predictors in the variable selection section first.")
else:
    # Controls for the two predictor sets. An empty reduced set means an
    # intercept-only model, which turns this into the overall F-test.
    reduced_vars = st.multiselect("Reduced model predictors", predictors,
                                  key="f_reduced")
    full_vars = st.multiselect("Full model predictors", predictors,
                               default=predictors, key="f_full")
    f_alpha = st.selectbox("Significance level", [0.01, 0.05, 0.10],
                           index=1, key="f_alpha")

    if st.button("Compare nested models"):
        # Variable-level check: every reduced predictor must be in the full model
        missing = [v for v in reduced_vars if v not in full_vars]
        bad_type = [v for v in full_vars
                    if v not in categorical and v not in numeric]

        if missing:
            st.error(f"The full model must contain every reduced predictor. "
                     f"Missing: {missing}")
        elif set(reduced_vars) == set(full_vars):
            st.error("The two models are identical. Add a predictor to the full model.")
        elif bad_type:
            st.error(f"These text columns must be marked categorical: {bad_type}")
        else:
            # Identical complete-case rows for BOTH models:
            # drop rows missing the response or ANY full-model predictor
            cc = data[[response] + full_vars].dropna().copy()
            y = cc[response].to_numpy(dtype=float)
            n_obs = len(cc)

            def build_design(frame, vars_):
                """Intercept + numeric columns + ALL k-1 dummies per categorical."""
                cats = [v for v in vars_ if v in categorical]
                quants = [v for v in vars_ if v not in categorical]
                pieces = [pd.DataFrame({"Intercept": np.ones(len(frame))},
                                       index=frame.index)]
                if quants:
                    pieces.append(frame[quants].astype(float))
                if cats:
                    pieces.append(pd.get_dummies(frame[cats].astype(str),
                                                 drop_first=True, dtype=float))
                return pd.concat(pieces, axis=1)

            Xr = build_design(cc, reduced_vars).to_numpy()
            Xf = build_design(cc, full_vars).to_numpy()

            rank_r = np.linalg.matrix_rank(Xr)
            rank_f = np.linalg.matrix_rank(Xf)
            rank_both = np.linalg.matrix_rank(np.hstack([Xr, Xf]))

            if rank_both != rank_f:
                # Column space of reduced is NOT inside column space of full
                st.error("The reduced model's column space is not contained "
                         "in the full model's. The models are not nested.")
            elif rank_f < Xf.shape[1]:
                st.error("The full design matrix is rank deficient. "
                         "Check for redundant or perfectly correlated predictors.")
            elif n_obs <= rank_f:
                st.error("Not enough complete observations for this model.")
            elif rank_f - rank_r <= 0:
                st.error("The full model adds no new independent columns.")
            else:
                def fit_rss(X):
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    res = y - X @ beta
                    return float(res @ res)

                rss_r, rss_f = fit_rss(Xr), fit_rss(Xf)
                q = rank_f - rank_r            # number of extra columns tested
                df_full = n_obs - rank_f       # residual df of the full model
                F_stat = ((rss_r - rss_f) / q) / (rss_f / df_full)
                p_val = stats.f.sf(F_stat, q, df_full)
                f_crit = stats.f.ppf(1 - f_alpha, q, df_full)

                st.write("Complete-case rows used:", n_obs)
                st.dataframe(pd.DataFrame({
                    "Model": ["Reduced", "Full"],
                    "Columns (rank)": [rank_r, rank_f],
                    "Residual df": [n_obs - rank_r, df_full],
                    "RSS": [rss_r, rss_f]
                }))
                st.write("Added predictors:",
                         [v for v in full_vars if v not in reduced_vars])
                st.write("Numerator df (q):", q, "| Denominator df:", df_full)
                st.write("F-statistic:", round(F_stat, 4))
                st.write("p-value:", round(p_val, 6))
                st.write("Critical F:", round(f_crit, 4))

                if p_val < f_alpha:
                    st.success(f"Reject H0 at alpha = {f_alpha}. The added "
                               "predictors significantly improve the model.")
                else:
                    st.info(f"Fail to reject H0 at alpha = {f_alpha}. No "
                            "significant improvement from the added predictors.")

                # Plot the F distribution with the rejection region
                xs = np.linspace(0.001, max(f_crit * 2, F_stat * 1.2), 500)
                fig_f, ax_f = plt.subplots(figsize=(8, 4))
                ax_f.plot(xs, stats.f.pdf(xs, q, df_full), color="navy",
                          label=f"F distribution (df = {q}, {df_full})")

                # Shade the rejection region
                ax_f.fill_between(xs, stats.f.pdf(xs, q, df_full),
                                  where=xs >= f_crit, color="red", alpha=0.25,
                                  label=f"Rejection region (alpha = {f_alpha})")    

                # Critical limit (dashed red line)
                ax_f.axvline(f_crit, color="red", linestyle="--",
                             label=f"Critical F = {f_crit:.3f}")

                # Observed test statistic (solid green line)
                ax_f.axvline(F_stat, color="green", linewidth=2.5,
                             label=f"Observed F = {F_stat:.3f}")

                ax_f.set(xlabel="F", ylabel="Density",
                         title="Nested F-test")
                ax_f.legend()
                ax_f.grid(alpha=0.2)
                st.pyplot(fig_f)
                plt.close(fig_f)
                
# Part 12: 10-fold cross-validation
st.subheader("Part 12: 10-fold cross-validation")

cv_vars = st.multiselect("Predictors for CV", predictors,
                         default=predictors, key="cv_vars")
degrees = st.multiselect("Polynomial degrees to compare", [1, 2, 3, 4, 5],
                         default=[1, 2, 3], key="cv_degrees")

# CHECKPOINT 12 CUSTOMIZATION: user-controlled KFold parameters
cv_shuffle = st.checkbox("Shuffle rows before splitting", value=True,
                         key="cv_shuffle")
cv_seed = st.number_input("Random state (seed)", min_value=0,
                          max_value=100000, value=1, step=1, key="cv_seed",
                          disabled=not cv_shuffle)

if st.button("Run 10-fold CV"):
    cv_quant = [v for v in cv_vars if v not in categorical]
    cv_cat = [v for v in cv_vars if v in categorical]
    bad_cv = [v for v in cv_quant if v not in numeric]

    if not cv_vars or not degrees:
        st.warning("Choose at least one predictor and one degree.")
    elif bad_cv:
        st.error(f"These text columns must be marked categorical: {bad_cv}")
    else:
        # Same complete-case rows for every candidate model
        cc_cv = data[[response] + cv_vars].dropna().reset_index(drop=True)

        if len(cc_cv) < 10:
            st.error("Need at least 10 complete rows for 10-fold CV.")
        else:
            # Build the folds ONCE so every candidate model uses the same splits.
            # random_state must be None when shuffle is off (sklearn requires it).
            kf = KFold(n_splits=10, shuffle=cv_shuffle,
                       random_state=int(cv_seed) if cv_shuffle else None)
            folds = list(kf.split(cc_cv))

            # Without numeric predictors, polynomial degree has no effect
            use_degrees = sorted(degrees) if cv_quant else [1]

            rows, all_fold_mse = [], {}
            for deg in use_degrees:
                fold_mse = []
                for train_idx, test_idx in folds:
                    train = cc_cv.iloc[train_idx]
                    test = cc_cv.iloc[test_idx]
                    tr_parts, te_parts = [], []

                    if cv_quant:
                        # Fit the transformer on TRAINING rows only
                        poly = PolynomialFeatures(degree=deg, include_bias=False)
                        tr_parts.append(poly.fit_transform(train[cv_quant]))
                        te_parts.append(poly.transform(test[cv_quant]))
                    if cv_cat:
                        enc = OneHotEncoder(drop="first", sparse_output=False,
                                            handle_unknown="ignore")
                        tr_parts.append(enc.fit_transform(train[cv_cat].astype(str)))
                        te_parts.append(enc.transform(test[cv_cat].astype(str)))

                    m = LinearRegression().fit(np.hstack(tr_parts),
                                               train[response])
                    pred = m.predict(np.hstack(te_parts))
                    fold_mse.append(mean_squared_error(test[response], pred))

                all_fold_mse[deg] = fold_mse
                rows.append({"Degree": deg,
                             "Mean CV MSE": np.mean(fold_mse),
                             "SD across folds": np.std(fold_mse, ddof=1),
                             "CV RMSE": np.sqrt(np.mean(fold_mse))})

            st.session_state["cv_results"] = dict(
                table=pd.DataFrame(rows), n=len(cc_cv),
                shuffle=cv_shuffle, seed=int(cv_seed))

if "cv_results" in st.session_state:
    cvr = st.session_state["cv_results"]
    st.write("Rows used:", cvr["n"],
             "| Shuffle:", cvr["shuffle"],
             "| Random state:", cvr["seed"] if cvr["shuffle"] else "None")
    st.dataframe(cvr["table"].style.format({
        "Mean CV MSE": "{:.4f}", "SD across folds": "{:.4f}",
        "CV RMSE": "{:.4f}"}))

    best = cvr["table"].loc[cvr["table"]["Mean CV MSE"].idxmin(), "Degree"]
    st.success(f"Lowest mean CV MSE: degree {int(best)}")
