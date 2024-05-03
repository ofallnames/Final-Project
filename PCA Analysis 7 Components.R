# Load necessary library
library(readr)

# Load the Excel file
myData <- read_csv("train.csv")

dev.off()
for (i in 1:length(myData)) {
  boxplot(myData[,i], main=names(myData[i]), type="l")
  
}

View(cor(myData[,-1]))
myData.st <- scale(myData[ , -1])
pca <- prcomp(myData.st)
summary(pca)

#display the weights for original variables in each component
View(pca$rotation)

#review the principal component score
View(pca$x)

#combine the original data with the obtained PCA scores
newData <- data.frame(myData, pca$x)
newData <- newData[ , -(2:14)]
head(newData)
screeplot(pca, type = "lines")

#Due to the summary from our screenplot we will 7 principal components
newData <- newData[ , -(10:16)]
write_csv(newData, "pcaAnalysis.csv")
