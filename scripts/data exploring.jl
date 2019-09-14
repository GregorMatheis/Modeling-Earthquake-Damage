# data exploring
# import Pkg
# Pkg.add("MLJ")
# Pkg.add("MLJModels")

#using Weave 
#weave("./reports/base_exploration.jmd", "md2html") 
import Pkg
for x in ["MLJ", "MLJModels", "CategoricalArrays", "CSV"] #DataFrames", "Gadfly", "Statistics", "StatsBase", 
    Pkg.add(x)
end
using CSV, DataFrames, Gadfly, Statistics, StatsBase, MLJ, MLJModels, CategoricalArrays

train_values = CSV.read("./Data/train_values.csv") |> DataFrame
train_labels = CSV.read("./Data/train_labels.csv") |> DataFrame

describe(train_values)
train_dt = join(train_values, train_labels, on = :building_id, kind = :inner)


cor_dt = train_dt[eltypes(train_dt) .!= Union{Missing, String}]
cormatrix = zeros(Float64, size(cor_dt,2), size(cor_dt,2))

for (i, x) in enumerate(eachcol(cor_dt))
    for (j, y) in enumerate(eachcol(cor_dt))
        cormatrix[i,j] = try cor(x[2], y[2]); catch; 0 end
    end
end
cormatrix = hcat(names(cor_dt), cormatrix)
plotdt = stack(DataFrame(cormatrix,vcat(:name, names(cor_dt))), 2:33, 1)

plot(plotdt,
    x = :variable,
    y = :name,
    color = :value,
    Geom.rectbin) |>
    SVGJS(10inch, 10inch)

plot(train_dt,
    x = :damage_grade,
    Geom.histogram)|>
    SVGJS(7inch, 7inch)

plot(train_dt,
    x = :age,
    Geom.histogram)|>
    SVGJS(7inch, 7inch)
train_dt[train_dt.age .> 900,:]

by(train_dt, :damage_grade, :age => mean)
plot(train_dt,
    x = :age,
    y = :damage_grade,
    Geom.density2d) |>
    SVGJS(7inch, 7inch)


categorical!(train_dt, :damage_grade)
X = DataFrame([try float(x); catch; x end for x = eachcol(train_dt[2:8])], names(train_dt[2:8]));
z = train_dt[:,end];

X = train_dt[2:(end-1)]

mutable struct WrappedBoost <: DeterministicNetwork
    boost_machine
end
@load XGBoostClassifier
function MLJ.fit(model::WrappedBoost, X, y)
    Xs = source(X)

    oneHot = machine(OneHotEncoder(), Xs)
    W = transform(oneHot, Xs)

    boost_class = model.boost_machine
    boost_machine = machine(boost_class, W, y)

    yhat = predict(boost_machine, W)
    fit!(yhat, verbosity=0)
    return yhat
end

wrapped_model = WrappedBoost(XGBoostClassifier(
    booster = "gbtree", #gblinear
    num_round = 10, #1
    disable_default_eval_metric = 0,
    eta = 0.3, #.3
    gamma = 0.001, # 0.0
    max_depth = 20, #6
    min_child_weight = 1.0,
    max_delta_step = 0.0,
    subsample = .8, #1.
    colsample_bytree = 1.0,
    colsample_bylevel = 1.0,
    lambda = 1.0,
    alpha = 0.0,
    tree_method = "auto",
    sketch_eps = 0.03,
    scale_pos_weight = 1.0,
    updater = "grow_colmaker",
    refresh_leaf = 1,
    process_type = "default",
    grow_policy = "depthwise",
    max_leaves = 0,
    max_bin = 256,
    predictor = "cpu_predictor",
    sample_type = "uniform",
    normalize_type = "tree",
    rate_drop = 0.0,
    one_drop = 0,
    skip_drop = 0.0,
    feature_selector = "cyclic",
    top_k = 0,
    tweedie_variance_power = 1.5,
    objective = "automatic", #"multi:softmax",
    base_score = 0.5,
    eval_metric = "mlogloss",
    seed = 0))

zz = CategoricalArray(string.(z))
mach = machine(wrapped_model, X, zz)#parse.(Int,string.(z)))
evaluate!(mach, resampling=Holdout(fraction_train=0.7, shuffle=true), measure=misclassification_rate)

test_values = CSV.read("./Data/test_values.csv") |> DataFrame
categorical!(test_values)
yhat = predict(mach, test_values[2:end])
yhat

plot(train_dt, x = :damage_grade, Geom.histogram)