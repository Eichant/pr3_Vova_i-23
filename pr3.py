import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OrdersAnalyzer:
    def __init__(self):
        self.data = None
        self.olap_cube = None
        
    def generate_data(self):
        customers = [
            {'CustomerID': 1, 'LastName': 'Іванов', 'City': 'Київ', 'Country': 'Україна'},
            {'CustomerID': 2, 'LastName': 'Петров', 'City': 'Львів', 'Country': 'Україна'},
            {'CustomerID': 3, 'LastName': 'Сидоров', 'City': 'Одеса', 'Country': 'Україна'},
            {'CustomerID': 4, 'LastName': 'Коваленко', 'City': 'Харків', 'Country': 'Україна'},
            {'CustomerID': 5, 'LastName': 'Шевченко', 'City': 'Дніпро', 'Country': 'Україна'},
            {'CustomerID': 6, 'LastName': 'Мельник', 'City': 'Вінниця', 'Country': 'Україна'},
            {'CustomerID': 7, 'LastName': 'Бондар', 'City': 'Запоріжжя', 'Country': 'Україна'},
            {'CustomerID': 8, 'LastName': 'Ткаченко', 'City': 'Житомир', 'Country': 'Україна'},
            {'CustomerID': 9, 'LastName': 'Кравченко', 'City': 'Полтава', 'Country': 'Україна'},
            {'CustomerID': 10, 'LastName': 'Олійник', 'City': 'Чернігів', 'Country': 'Україна'}
        ]
        self.customers_df = pd.DataFrame(customers)
        
        suppliers = [
            {'SupplierID': 101, 'Name': 'ТехноПостач', 'City': 'Київ', 'Country': 'Україна'},
            {'SupplierID': 102, 'Name': 'ЕлектронСервіс', 'City': 'Львів', 'Country': 'Україна'},
            {'SupplierID': 103, 'Name': 'ОдеськийПостач', 'City': 'Одеса', 'Country': 'Україна'},
            {'SupplierID': 104, 'Name': 'ПромМетал', 'City': 'Харків', 'Country': 'Україна'},
            {'SupplierID': 105, 'Name': 'МегаОпт', 'City': 'Дніпро', 'Country': 'Україна'},
            {'SupplierID': 106, 'Name': 'БудПостач', 'City': 'Вінниця', 'Country': 'Україна'},
            {'SupplierID': 107, 'Name': 'АгроПостач', 'City': 'Запоріжжя', 'Country': 'Україна'},
            {'SupplierID': 108, 'Name': 'ФармПостач', 'City': 'Житомир', 'Country': 'Україна'},
            {'SupplierID': 109, 'Name': 'ТекстильГруп', 'City': 'Полтава', 'Country': 'Україна'},
            {'SupplierID': 110, 'Name': 'ХімПостач', 'City': 'Чернігів', 'Country': 'Україна'}
        ]
        self.suppliers_df = pd.DataFrame(suppliers)
        
        np.random.seed(42)
        n_orders = 500
        
        orders_data = []
        start_date = datetime(2025, 1, 1)
        
        for i in range(1, n_orders + 1):
            customer = np.random.choice(self.customers_df['CustomerID'])
            supplier = np.random.choice(self.suppliers_df['SupplierID'])
            
            days_offset = np.random.randint(0, 365)
            execution_date = start_date + timedelta(days=days_offset)
            
            quantity = np.random.randint(1, 100)
            price = np.random.uniform(50, 5000)
            total = quantity * price
            
            orders_data.append({
                'OrderID': i,
                'CustomerID': customer,
                'SupplierID': supplier,
                'ExecutionDate': execution_date,
                'Quantity': quantity,
                'Price': round(price, 2),
                'Total': round(total, 2)
            })
        
        orders_df = pd.DataFrame(orders_data)
        orders_df = pd.merge(orders_df, self.customers_df, on='CustomerID')
        orders_df = pd.merge(orders_df, self.suppliers_df, on='SupplierID', 
                           suffixes=('_Customer', '_Supplier'))
        
        # Вручну перейменуємо колонку Name
        if 'Name' in orders_df.columns:
            orders_df = orders_df.rename(columns={'Name': 'SupplierName'})
        
        orders_df['Year'] = orders_df['ExecutionDate'].dt.year
        orders_df['Month'] = orders_df['ExecutionDate'].dt.month
        orders_df['MonthName'] = orders_df['ExecutionDate'].dt.month_name()
        orders_df['Week'] = orders_df['ExecutionDate'].dt.isocalendar().week
        orders_df['DayOfWeek'] = orders_df['ExecutionDate'].dt.day_name()
        
        self.data = orders_df
        print(f"Згенеровано {len(self.data)} замовлень")
        return self.data
    
    def build_cube(self):
        # Використовуємо правильні назви колонок
        column_names = [
            'ExecutionDate', 'Year', 'Month', 'MonthName', 'Week', 'DayOfWeek',
            'LastName', 'City_Customer', 'Country_Customer',
            'Quantity', 'Price', 'Total'
        ]
        
        # Додаємо назву постачальника (може бути 'SupplierName' або 'Name')
        if 'SupplierName' in self.data.columns:
            column_names.insert(9, 'SupplierName')
            column_names.extend(['City_Supplier', 'Country_Supplier'])
        elif 'Name' in self.data.columns:
            column_names.insert(9, 'Name')
            column_names.extend(['City_Supplier', 'Country_Supplier'])
        else:
            # Якщо жодної з цих колонок немає, шукаємо інші варіанти
            for col in self.data.columns:
                if 'Supplier' in col and ('Name' in col or 'City' in col or 'Country' in col):
                    column_names.append(col)
        
        self.olap_cube = self.data[column_names]
        
        print(f"Куб: {self.olap_cube.shape}")
        print("Колонки куба:", self.olap_cube.columns.tolist())
        return self.olap_cube
    
    def analyze(self):
        # Визначаємо назву колонки з ім'ям постачальника
        supplier_name_col = None
        for col in self.olap_cube.columns:
            if col in ['SupplierName', 'Name', 'Name_Supplier']:
                supplier_name_col = col
                break
        
        if supplier_name_col is None:
            supplier_name_col = self.olap_cube.columns[9]  # Беремо 10-ту колонку
        
        price_analysis = pd.pivot_table(
            self.olap_cube,
            values='Price',
            index=['LastName', 'Country_Customer'],
            columns=[supplier_name_col, 'Country_Supplier'],
            aggfunc=['mean', 'min', 'max'],
            fill_value=0
        )
        
        quantity_analysis = pd.pivot_table(
            self.olap_cube,
            values='Quantity',
            index=['MonthName', 'City_Customer'],
            columns=['City_Supplier'],
            aggfunc=['sum', 'mean'],
            fill_value=0
        )
        
        total_by_country = pd.pivot_table(
            self.olap_cube,
            values='Total',
            index=['Country_Customer'],
            columns=['Country_Supplier'],
            aggfunc='sum',
            fill_value=0
        )
        
        return price_analysis, quantity_analysis, total_by_country
    
    def visualize(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        monthly_total = self.olap_cube.groupby('MonthName')['Total'].sum()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_total = monthly_total.reindex(month_order)
        
        axes[0, 0].bar(monthly_total.index, monthly_total.values / 1000, color='skyblue')
        axes[0, 0].set_title('Сума по місяцях')
        axes[0, 0].set_ylabel('Тис. грн')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        orders_by_customer = self.olap_cube.groupby('LastName')['Quantity'].sum().nlargest(8)
        axes[0, 1].bar(orders_by_customer.index, orders_by_customer.values, color='lightcoral')
        axes[0, 1].set_title('Клієнти за кількістю')
        axes[0, 1].set_ylabel('Кількість')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Знаходимо колонку з ім'ям постачальника для графіка
        supplier_col = None
        for col in self.olap_cube.columns:
            if any(x in col for x in ['Supplier', 'Name']) and col != 'LastName':
                supplier_col = col
                break
        
        if supplier_col:
            avg_price_supplier = self.olap_cube.groupby(supplier_col)['Price'].mean().sort_values()
            axes[0, 2].barh(avg_price_supplier.index, avg_price_supplier.values, color='lightgreen')
            axes[0, 2].set_title('Середня ціна постачальників')
            axes[0, 2].set_xlabel('Ціна, грн')
            axes[0, 2].grid(axis='x', alpha=0.3)
        
        city_dist = self.olap_cube.groupby('City_Customer')['Total'].sum()
        axes[1, 0].pie(city_dist.values, labels=city_dist.index,
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Сума по містах клієнтів')
        
        monthly_price = self.olap_cube.groupby('MonthName')['Price'].mean()
        monthly_price = monthly_price.reindex(month_order)
        
        axes[1, 1].plot(monthly_price.index, monthly_price.values, 
                       marker='o', color='purple', linewidth=2)
        axes[1, 1].set_title('Ціна по місяцях')
        axes[1, 1].set_ylabel('Ціна, грн')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        heatmap_data = pd.pivot_table(
            self.olap_cube,
            values='Total',
            index=['Country_Customer'],
            columns=['Country_Supplier'],
            aggfunc='sum',
            fill_value=0
        )
        
        im = axes[1, 2].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
        axes[1, 2].set_title('Сума: Країна клієнта × Країна постачальника')
        axes[1, 2].set_xticks(range(len(heatmap_data.columns)))
        axes[1, 2].set_yticks(range(len(heatmap_data.index)))
        axes[1, 2].set_xticklabels(heatmap_data.columns, rotation=45)
        axes[1, 2].set_yticklabels(heatmap_data.index)
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 8))
        numeric_cols = ['Quantity', 'Price', 'Total']
        correlation = self.olap_cube[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('Кореляція між показниками')
        plt.tight_layout()
        plt.savefig('correlation.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def export(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'results_{timestamp}.xlsx'
        
        # Знаходимо назву колонки постачальника для статистики
        supplier_col = None
        for col in self.olap_cube.columns:
            if any(x in col for x in ['Supplier', 'Name']) and col != 'LastName':
                supplier_col = col
                break
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self.olap_cube.to_excel(writer, sheet_name='Data', index=False)
            
            price, quantity, total = self.analyze()
            price.to_excel(writer, sheet_name='Price_Analysis')
            quantity.to_excel(writer, sheet_name='Quantity_Analysis')
            total.to_excel(writer, sheet_name='Total_by_Country')
            
            stats_data = {
                'Показник': [
                    'Кількість замовлень',
                    'Загальна сума',
                    'Середня кількість',
                    'Середня ціна',
                    'Кількість клієнтів'
                ],
                'Значення': [
                    len(self.olap_cube),
                    f"{self.olap_cube['Total'].sum():,.2f} грн",
                    f"{self.olap_cube['Quantity'].mean():.1f} од.",
                    f"{self.olap_cube['Price'].mean():.2f} грн",
                    self.olap_cube['LastName'].nunique()
                ]
            }
            
            if supplier_col:
                stats_data['Показник'].append('Кількість постачальників')
                stats_data['Значення'].append(self.olap_cube[supplier_col].nunique())
            
            stats_data['Показник'].append('Період')
            stats_data['Значення'].append(
                f"{self.olap_cube['ExecutionDate'].min().date()} - {self.olap_cube['ExecutionDate'].max().date()}"
            )
            
            stats = pd.DataFrame(stats_data)
            stats.to_excel(writer, sheet_name='Stats', index=False)
        
        print(f"Експортовано: {filename}")
        return filename

def main():
    analyzer = OrdersAnalyzer()
    analyzer.generate_data()
    analyzer.build_cube()
    analyzer.analyze()
    analyzer.visualize()
    analyzer.export()
    
    print("Готово")

if __name__ == "__main__":
    main()