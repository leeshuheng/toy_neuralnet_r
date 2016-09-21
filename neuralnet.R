### 2016年 09月 19日 星期一 14:37:43 CST #############
### author: 李小丹(Li Shao Dan) 字 殊恒(shuheng)
### K.I.S.S
### S.P.O.T


sigmoid <- function(x) {
	return(1 / (1 + exp(-x)));
}

sigmoid.df <- function(z) {
	return(z / (1 + z) ^ 2)
}

tanh.df <- function(z) {
	return(1 - z ^ 2)
}

# sigmoid.fun <- list(fun = sigmoid, df = sigmoid.df)
# tanh.fun <- list(fun = tanh, df = tanh.df)

weight.matrix <- function(input, hidden, output) {
	l <- length(hidden)
	p = input + 1
	res <- list()
	for(i in 1:l) {
		r <- hidden[i]
		res[[i]] <- matrix(runif(p * r), nrow = p, byrow = T)
		p <- r + 1;
	}
	res[[l+1]] <- matrix(runif(p * output), nrow = p, byrow = T)
	return(res)
}

error.fun <- function(t, y) {
	#print(str(t))
	#print("================")
	#print(str(y))
	#print("================")
	return(sum(0.5 * (t - y) ^ 2))
}

error.df <- function(t, y) {
	return((t - y))
}

.feedforward <- function(x, w, acf) {
	#print(x)
	#print("================")
	#print(w)
	#print("================")
	return(acf(x %*% w))
}

feedforward <- function(x, w, act.fun) {
	m <- x
	f <- list()
	facf <- list()
	acf <- sigmoid
	if(act.fun == "tanh") acf <- tanh
	for(i in 1:length(w)) {
		m <- cbind(m, rep(1, nrow(x)))
		f[[i]] <- m

		#print(m)
		#print(w[[i]])
		m <- .feedforward(m, w[[i]], acf)
		facf[[i]] <- m
		#print(m)
		#print("================")
	}
	return(list(f = f, facf = facf))
}

cal.grad <- function(df, node) {
	tmp.df <- NULL
	facf <- node[["facf"]]
	f <- node[["f"]]
	#print(ncol(df))

	for(i in 1:ncol(facf)) {
		for(j in 1:(ncol(f))) {
			tmp <- facf[,i, drop = F] * f[,j, drop = F]
			ifelse(is.null(tmp.df),
				   tmp.df <- matrix(tmp, ncol = 1),
				   tmp.df <- cbind(tmp.df, tmp))
		}
	}
	res <- NULL
# 	for(i in 1:ncol(df)) {
# 		for(j in 1:ncol(tmp.df)) {
# 			tmp <- df[,i, drop = F] * tmp.df[,j, drop = F]
# 			ifelse(is.null(res),
# 				   res <- matrix(tmp, ncol = 1),
# 				   res <- cbind(res, tmp))
# 		}
# 	}
	for(i in 1:ncol(tmp.df)) {
		tmp <- matrix(rep(0, nrow(tmp.df)), ncol = 1)

		for(j in 1:ncol(df))
			tmp <- tmp + tmp.df[,i, drop = F] * df[, j, drop = F]

		ifelse(is.null(res),
			   res <- tmp,
			   res <- cbind(res, tmp))
	}
	return(res)
}

recal.weight <- function(wm, df, delta) {
	df <- apply(df, 2, sum)
	wm <- wm - delta * df
	return(wm)
}

.backpropagation <- function(wm, nn, y, acf.df, delta) {
	l <- length(nn[["facf"]])
	df <- error.df(nn[["facf"]][[l]], y)

	for(i in l:1) {
		node <- list(facf = nn[["facf"]][[i]], f = nn[["f"]][[i]])
		rm.col <- ncol(node[["f"]]) * 1:ncol(df)

		df <- cal.grad(df, node)
		wm[[i]] <- recal.weight(wm[[i]], df, delta)

		df <- df[,-rm.col, drop = F]
		#print(wm[[i]])
	}
	return(wm)
}


backpropagation <- function(wm, nn, y, acf, delta) {
	acf.df <- sigmoid.df
	if(acf == "tanh") acf.df <- tanh.df

	return(.backpropagation(wm, nn, y, acf.df, delta))
}


neural.net <- function(x, y, hidden = 1, delta = 0.5,
					   iter.max = 1000L, act.fun = "sigmoid",
					   threshold = 0.001) {
	if(class(x) != "matrix") stop("x must be a matrix!")

	input <- ncol(x)
	output <- ncol(y)
	wm <- weight.matrix(input, hidden, output)
	#x <- cbind(x, rep(1, nrow(x)))

	for(i in 1:iter.max) {

		nn <- feedforward(x, wm, act.fun)

		(error <- error.fun(nn[["facf"]][[length(nn[["facf"]])]],  y))
		if(mean(abs(error) < threshold)) break

		wm <- backpropagation(wm, nn, y, act.fun, delta)
		#print(i)
	}
	return(list(coef = wm, acf = act.fun, input = input))
}

predict.nn <- function(fit, x) {
	if(class(x) != "matrix") stop("error")

	wm <- fit[["coef"]]
	act.fun <- fit[["acf"]]
	input <- fit[["input"]]

	if(ncol(x) != input) stop("error")

	nn <- feedforward(x, wm, act.fun)
	return(nn[["facf"]][[length(nn)]])
}

test.nn <- function() {
	d.f <- data.frame(a = rnorm(10, 1), b = rnorm(10, 5))
	d.f <- rbind(d.f, data.frame(a = rnorm(10, 5), b = rnorm(10, 10)))

	y <- matrix(c(rep(0, 10), rep(1, 10)))
	x <- data.matrix(d.f)
	x <- scale(x)

	rm(d.f)

	mean(x[1:10, 1])
	mean(x[1:10, 2])
	mean(x[11:20, 1])
	mean(x[11:20, 2])

	(fit <- neural.net(x, y, hidden = 3, delta = 0.5))
	predict.nn(fit, x)
	round(predict.nn(fit, x))
}

test.nn()
