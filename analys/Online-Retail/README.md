dataset ----> https://www.kaggle.com/datasets/hellbuoy/online-retail-customer-clustering


# Customer Segmentation using K-Means and Hierarchical Clustering

This project performs customer segmentation using the **Online Retail** dataset by applying both **K-Means** and **Hierarchical Clustering** methods. The dataset includes customer purchase information and product details.

## Project Overview

The project uses customer-level aggregated data to cluster customers based on their purchasing behavior. Two clustering techniques are implemented:
- **K-Means Clustering**
- **Hierarchical Clustering**

### Dataset Columns:
- `InvoiceNo`: Invoice number
- `StockCode`: Product code
- `Description`: Product description
- `Quantity`: Quantity of products purchased
- `InvoiceDate`: Date of invoice
- `UnitPrice`: Price per unit of the product
- `CustomerID`: Unique customer identifier
- `Country`: Customer's country

## Steps:
1. **Data Preprocessing**:
   - Removing missing values.
   - Filtering out rows with invalid quantities or prices.
   - Aggregating data by `CustomerID` to create summary statistics.
   
2. **Clustering**:
   - **K-Means Clustering**: Optimal number of clusters determined using the elbow method.
   - **Hierarchical Clustering**: Optimal number of clusters visualized using a dendrogram.
   
3. **Visualization**:
   - Scatter plots to visualize the clusters based on customer purchase quantities and average unit prices.

## How to Run:
1. Clone the repository:
   ```bash
   git clone <repository_url>

   pip install -r requirements.txt

   python main.py

   ```


## Visualization

K-Means Clustering results and optimal number of clusters are visualized using the elbow method.
Hierarchical Clustering results are visualized with a dendrogram.
