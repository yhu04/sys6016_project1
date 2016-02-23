######################################################
# Machine Learning  - Titanic Project
# Preprocessing data
######################################################

#Converts categorical strings in titanic dataset to numerical data
#Input: Titanic dataset
#Output: numerical titanic data
convert_to_numeric = function(titanic) {
  
  titanic.numeric = subset(titanic, select = -c(name, ticket, cabin, boat, home.dest))
  
  # sex: female = 0, male = 1
  titanic.numeric$sex = gsub("female", 0, titanic.numeric$sex)
  titanic.numeric$sex = gsub("male", 1, titanic.numeric$sex)
  
  # embarked: blank = 0, C = 1, Q = 2, S = 3
  titanic.numeric$embarked = gsub("C", 1, titanic.numeric$embarked)
  titanic.numeric$embarked = gsub("Q", 2, titanic.numeric$embarked)
  titanic.numeric$embarked = gsub("S", 3, titanic.numeric$embarked)
  
  # body: found 1, NA 0
  titanic.numeric$body[is.na(titanic.numeric$body)] <- 0
  titanic.numeric[which(titanic.numeric$body != 0), 9] <- 1
  
  # create age.bin column: <5 = 0, <20 = 1, >20 && <40 = 2, >40 = 3
  age.bin = matrix(nrow = nrow(titanic), ncol = 1)
  for (i in 1:nrow(titanic)) {
    age = titanic$age[i]
    if (is.na(age)) {
      next
    } else if (age <= 5) {
      age.bin[i] = 0
    } else if (age <= 20) {
      age.bin[i] = 1
    } else if (age <= 40) {
      age.bin[i]= 2
    } else {
      age.bin[i] = 3
    }
  }
  titanic.numeric = cbind(titanic.numeric, age.bin)
  
  titanic.numeric$sex = as.numeric(titanic.numeric$sex)
  titanic.numeric$embarked = as.numeric(titanic.numeric$embarked)
  titanic.numeric$body = as.numeric(titanic.numeric$body)
  
  return(titanic.numeric)
}

# Library imports
library(DMwR)
library(gdata)


# Per-user configuration
# excel_path <- "/Users/zverham/Documents/School/Masters/SemesterTwo/SYS6016/sys6016_project1/data/titanic3.xls"
# csv_path <-  "/Users/zverham/Documents/School/Masters/SemesterTwo/SYS6016/sys6016_project1/data/titanic_preprocessed.csv"
excel_path <- "~/Desktop/UVA Spring 2016/Machine Learning | SYS 6016/HW/Titanic Project/titanic3.xls"
csv_path <- "~/Desktop/UVA Spring 2016/Machine Learning | SYS 6016/HW/Titanic Project/titanic_preprocessed.csv"

# Load data
titanic = read.xls(excel_path)

summary (survived ~ age + sex + pclass + sibsp + parch, data=titanic)

# count missing data in each column
# age: 263 missing
# fare: 1
# body: 1188

summary(titanic$embarked)

# create matrix of only the numeric data
titanic.numeric = convert_to_numeric(titanic)
titanic.knn = knnImputation(titanic.numeric, k=10) # imputes 263 age values, 1 fare, 2 embarked
sum(is.na(titanic.knn))

# colSums(is.na(titanic)) 
# colSums(is.na(titanic.numeric)) 
# colSums(is.na(titanic.knn)) 

colSums(is.na(titanic.knn)) 
summary(titanic.numeric$embarked)

symnum(cor(titanic.knn, use = "complete.obs"))

summary(titanic.knn$fare)

write.csv(titanic.knn, csv_path, row.names=FALSE)

