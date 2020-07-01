bookdown::serve_book(dir = ".", output_dir = "_book", preview = TRUE, in_session = TRUE)

# shell.exec(paste0(getwd(), '/_book/index.html' ) )


# 
# ```{r, eval=TRUE, echo=FALSE, fig.cap='Instalation of package from local files', out.width='500px'}
# knitr::include_graphics('figures/git_repo_Cloning.png')
# ```

# Inline-style: 
#   ![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")




library(gridExtra)
library(grid)
d <- head(iris[,1:3])
grid.table(d)

d[2,3] <- "this is very wwwwwide"
d[1,2] <- "this\nis\ntall"
colnames(d) <- c("alpha*integral(xdx,a,infinity)",
                 "this text\nis high", 'alpha/beta')

tt <- ttheme_default(colhead=list(fg_params = list(parse=TRUE)))

tt3 <- ttheme_default(core=list(fg_params=list(hjust=0, x=0.1)),
                      rowhead=list(fg_params=list(hjust=0, x=0)))

grid.table(t, theme = tt3)

theme=tt1





ttheme_default(core=list(fg_params=list(hjust=0, x=0.1)),
               rowhead=list(fg_params=list(hjust=0, x=0)))
grid.arrange(
  tableGrob(t, theme=tt1))

grid.table(t)
