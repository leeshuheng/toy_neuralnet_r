### 2016年 09月 19日 星期一 14:37:43 CST #############
### author: 李小丹(Li Shao Dan) 字 殊恒(shuheng)
### K.I.S.S
### S.P.O.T


sigmoid <- function(x) {
	return(1 / (1 + exp(-x)));
}

sigmoid.df <- function(z) {
	return(z * (1 - z))
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
		#res[[i]] <- matrix(runif(p * r), nrow = p, byrow = T)
		res[[i]] <- matrix(rnorm(p * r, sd = 0.5), nrow = p, byrow = T)
		p <- r + 1;
	}
	#res[[l+1]] <- matrix(runif(p * output), nrow = p, byrow = T)
	res[[l+1]] <- matrix(rnorm(p * output, sd = 0.5), nrow = p, byrow = T)
	return(res)
}

error.fun <- function(t, y) {
	#print(str(t))
	#print("================")
	#print(str(y))
	#print("================")
	#print(t[,-ncol(t), drop = F])
	return((0.5 * (y - t[,-ncol(t), drop = F]) ^ 2))
}

error.df <- function(t, y) {
	return((t[,-ncol(t), drop = F] - y))
}

.feedforward <- function(x, w, acf) {
	#print(x)
	#print("================")
	#print(w)
	#print("================")
	return((x %*% w))
}

# feedforward <- function(x, w, act.fun) {
# 	m <- x
# 	f <- list()
# 	facf <- list()
# 	acf <- sigmoid
# 	if(act.fun == "tanh") acf <- tanh
# 	for(i in 1:length(w)) {
# 		m <- cbind(m, rep(1, nrow(x)))
# 		f[[i]] <- m
# 
# 		#print(m)
# 		#print(w[[i]])
# 		m <- .feedforward(m, w[[i]], acf)
# 		facf[[i]] <- m
# 		#print(m)
# 		#print("================")
# 	}
# 	return(list(f = f, facf = facf))
# }

feedforward <- function(x, w, act.fun) {
	f <- list()
	facf <- list()
	acf <- sigmoid
	if(act.fun == "tanh") acf <- tanh

	m <- cbind(x, rep(1, nrow(x)))
	f[[1]] <- NA
	facf[[1]] <- m
	l <- length(w)
	for(i in 1:l) {
		m <- m %*% w[[i]]
		f[[i+1]] <- m

		m <- acf(m)
		m <- cbind(m, rep(1, nrow(m)))
		facf[[i+1]] <- m
	}
	return(list(f = f, facf = facf))
}

select.df <- function(v) {
	return(v[which.max(abs(v))])
	#return(sample(v, 1))
	#return(mean(v))
}

recal.weight <- function(wm, df, nr, nc, delta) {
	df <- apply(df, 2, select.df)
	#print(nr)
	#print(nc)
	#print(nrow(wm))
	#print(ncol(wm))
	df <- matrix(df, byrow = T, nrow = nr, ncol = nc)
	#print(df)
	wm <- wm - delta * df
	return(wm)
}

# .backpropagation <- function(wm, nn, y, acf.df, delta) {
# 	l <- length(nn[["facf"]])
# 	df <- error.df(nn[["facf"]][[l]], y)
# 
# 	for(i in l:1) {
# 		node <- list(facf = nn[["facf"]][[i]], f = nn[["f"]][[i]])
# 		rm.col <- ncol(node[["f"]]) * 1:ncol(df)
# 
# 		df <- cal.grad(df, node)
# 		wm[[i]] <- recal.weight(wm[[i]], df, delta)
# 
# 		df <- df[,-rm.col, drop = F]
# 		#print(wm[[i]])
# 	}
# 	return(wm)
# }

# cal.grad <- function(df, node) {
# 	tmp.df <- NULL
# 	facf <- node[["facf"]]
# 	f <- node[["f"]]
# 	#print(ncol(df))
# 
# 	for(i in 1:ncol(facf)) {
# 		for(j in 1:(ncol(f))) {
# 			tmp <- facf[,i, drop = F] * f[,j, drop = F]
# 			ifelse(is.null(tmp.df),
# 				   tmp.df <- matrix(tmp, ncol = 1),
# 				   tmp.df <- cbind(tmp.df, tmp))
# 		}
# 	}
# 	res <- NULL
# 	for(i in 1:ncol(tmp.df)) {
# 		tmp <- matrix(rep(0, nrow(tmp.df)), ncol = 1)
# 
# 		for(j in 1:ncol(df))
# 			tmp <- tmp + tmp.df[,i, drop = F] * df[, j, drop = F]
# 
# 		ifelse(is.null(res),
# 			   res <- tmp,
# 			   res <- cbind(res, tmp))
# 	}
# 	return(res)
# }

cal.grad <- function(df, node2, node1, acf.df) {
	f <- node2[["facf"]]
	acfdf <- NULL
	for(i in 1:(ncol(f)-1)) {
		tmp <- acf.df(f[, i, drop = F])
		ifelse(is.null(acfdf),
			   acfdf <- matrix(tmp, ncol = 1),
			   acfdf <- cbind(acfdf, tmp))
	}
	fdf <- NULL
	for(i in 1:ncol(acfdf)) {
		tmp <- matrix(rep(0, nrow(df)), ncol = 1)
		for(j in 1:ncol(df))
			tmp <- tmp + df[,j, drop = F] * acfdf[,i, drop = F]
		ifelse(is.null(fdf),
			   fdf <- matrix(tmp, ncol = 1),
			   fdf <- cbind(fdf, tmp))
	}
	facf <- node1[["facf"]]
	res <- NULL
	for(i in 1:ncol(facf)) {
		for(j in 1:ncol(fdf)) {
			tmp <- fdf[,j, drop = F] * facf[,i, drop = F]
			ifelse(is.null(res),
				   res <- matrix(tmp, ncol = 1),
				   res <- cbind(res, tmp))
		}
	}
	return(res)
}

.backpropagation <- function(wm, nn, y, acf.df, delta) {
	l <- length(nn[["facf"]])
	df <- error.df(nn[["facf"]][[l]], y)
	for(i in l:2) {
		node2 <- list(f = nn[["f"]][[i]], facf = nn[["facf"]][[i]])
		node1 <- list(f = nn[["f"]][[i-1]], facf = nn[["facf"]][[i-1]])
		df <- cal.grad(df, node2, node1, acf.df)
		nr <- ncol(node1[["facf"]])
		nc <- ncol(node2[["facf"]]) - 1
		wm[[i-1]] <- recal.weight(wm[[i-1]], df, nr, nc, delta)
		df <- df[,1:((nr-1) * nc), drop = F]
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
	error <- 0
	#x <- cbind(x, rep(1, nrow(x)))

	for(i in 1:iter.max) {

		nn <- feedforward(x, wm, act.fun)

		(error <- error.fun(nn[["facf"]][[length(nn[["facf"]])]],  y))
		if((max(error) < threshold)) break

		wm <- backpropagation(wm, nn, y, act.fun, delta)
		#nn[["facf"]][[4]]
		#print(i)
	}
	if(max(error) > threshold) print("Not convergence")
	return(list(coef = wm, acf = act.fun, input = input,
				error = max(error), conv = max(error) < threshold))
}

predict.nn <- function(fit, x) {
	if(class(x) != "matrix") stop("error")

	wm <- fit[["coef"]]
	act.fun <- fit[["acf"]]
	input <- fit[["input"]]

	if(ncol(x) != input) stop("error")

	nn <- feedforward(x, wm, act.fun)
	res <- nn[["facf"]][[length(nn[["facf"]])]]
	return(res[,-ncol(res), drop = F])
}


test.nn <- function() {
	half.data <- 200

	d.f <- data.frame(a = rnorm(half.data, 1), b = rnorm(half.data, 2))
	d.f <- rbind(d.f, data.frame(a = rnorm(half.data, 5), b = rnorm(half.data, 7)))

	y <- matrix(c(rep(0, half.data), rep(1, half.data)), ncol = 1)
	x <- data.matrix(d.f)
	x <- scale(x)

	rm(d.f)

	(fit <- neural.net(x, y, hidden = c(2, 2, 2), delta = 0.5,
					   iter.max = 1000L, threshold = 0.1))
	print(predict.nn(fit, x))
	if(fit[["conv"]]) print("Convergence")
	res <- round(predict.nn(fit, x))
	print(fit)
	print(table(y, res))
}

test.nn()
