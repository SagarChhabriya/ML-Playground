import sqlite3
import streamlit as st
from datetime import datetime
import pandas as pd

# Initialize database with two tables


def init_db():
    try:
        conn = sqlite3.connect("inventory.sqlite")
        cursor = conn.cursor()

        # Products table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            current_stock INTEGER DEFAULT 0
        )
        """)

        # Movements table (audit log)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            type TEXT CHECK(type IN ('STOCK_IN', 'SALE', 'ADJUSTMENT')),
            quantity INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Error initializing database: {e}")

# Add new product


def add_product(name):
    try:
        conn = sqlite3.connect("inventory.sqlite")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO products (name) VALUES (?)", (name,))
        conn.commit()
        st.success(f"Added product: {name}")
    except sqlite3.IntegrityError:
        st.warning(f"Product '{name}' already exists!")
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

# Record movement and update stock


def record_movement(product_id, movement_type, quantity):
    try:
        conn = sqlite3.connect("inventory.sqlite")
        cursor = conn.cursor()

        # Record the movement
        cursor.execute("""
        INSERT INTO movements (product_id, type, quantity)
        VALUES (?, ?, ?)
        """, (product_id, movement_type, quantity))

        # Update current stock
        if movement_type == 'STOCK_IN':
            cursor.execute("""
            UPDATE products 
            SET current_stock = current_stock + ? 
            WHERE id = ?
            """, (quantity, product_id))
        else:  # SALE or ADJUSTMENT
            cursor.execute("""
            UPDATE products 
            SET current_stock = current_stock - ? 
            WHERE id = ?
            """, (quantity, product_id))

        conn.commit()
        st.success(f"Recorded {movement_type} of {quantity} units")

    except sqlite3.Error as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    finally:
        conn.close()

# View current inventory


def view_inventory():
    try:
        conn = sqlite3.connect("inventory.sqlite")
        cursor = conn.cursor()

        cursor.execute("""
        SELECT id, name, current_stock 
        FROM products
        ORDER BY name
        """)

        # Fetch the data and return as a pandas DataFrame
        products = cursor.fetchall()
        conn.close()
        return pd.DataFrame(products, columns=["ID", "Product", "Stock"])
    except sqlite3.Error as e:
        st.error(f"Error fetching inventory: {e}")
        return pd.DataFrame(columns=["ID", "Product", "Stock"])

# View movement history


def view_movements(product_id=None):
    try:
        conn = sqlite3.connect("inventory.sqlite")
        cursor = conn.cursor()

        if product_id:
            cursor.execute("""
            SELECT m.type, m.quantity, m.timestamp, p.name
            FROM movements m
            JOIN products p ON m.product_id = p.id
            WHERE m.product_id = ?
            ORDER BY m.timestamp DESC
            LIMIT 10
            """, (product_id,))
            title = f"Last 10 movements for product {product_id}"
        else:
            cursor.execute("""
            SELECT m.type, m.quantity, m.timestamp, p.name
            FROM movements m
            JOIN products p ON m.product_id = p.id
            ORDER BY m.timestamp DESC
            LIMIT 10
            """)
            title = "Last 10 movements across all products"

        # Fetch the data and return as a pandas DataFrame
        movements = cursor.fetchall()
        conn.close()
        return title, pd.DataFrame(movements, columns=["Type", "Qty", "Timestamp", "Product"])
    except sqlite3.Error as e:
        st.error(f"Error fetching movements: {e}")
        return "", pd.DataFrame(columns=["Type", "Qty", "Timestamp", "Product"])

# Delete product


def delete_product(product_id):
    try:
        conn = sqlite3.connect("inventory.sqlite")
        cursor = conn.cursor()

        # Verify product exists
        cursor.execute("SELECT name FROM products WHERE id = ?", (product_id,))
        product = cursor.fetchone()
        if not product:
            st.error("Product not found!")
            return

        # Delete the product and all its movements
        cursor.execute(
            "DELETE FROM movements WHERE product_id = ?", (product_id,))
        cursor.execute("DELETE FROM products WHERE id = ?", (product_id,))
        conn.commit()
        st.success(f"Deleted product {product_id} and all related records")
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    finally:
        conn.close()

# Streamlit app interface


def main():
    st.title("Inventory System")

    init_db()  # Ensure the database is initialized

    # Sidebar navigation
    menu = ["Add Product", "Stock In", "Record Sale",
            "View Inventory", "View Movements", "Delete Product"]
    choice = st.sidebar.selectbox("Choose an option", menu)

    # Add Product
    if choice == "Add Product":
        st.subheader("Add New Product")
        name = st.text_input("Product Name")
        if st.button("Add Product"):
            if name:
                add_product(name)
            else:
                st.warning("Please enter a product name!")

   # Stock In
    elif choice == "Stock In":
        st.subheader("Stock In")
        inventory = view_inventory()
        if inventory.empty:
            st.warning("No products available to stock in.")
        else:
            # Correctly create the product options dictionary
            product_options = {row["Product"]: row["ID"]
                               for _, row in inventory.iterrows()}

            # Select the product to stock in
            product_name = st.selectbox(
                "Select Product", inventory["Product"].tolist())

            # Get the product ID from the selected product name
            product_id = product_options.get(product_name)

            # Get the quantity to add to stock
            quantity = st.number_input("Quantity to add", min_value=1)

            # If button is pressed, record the stock-in
            if st.button("Record Stock In"):
                if product_name and quantity:
                    record_movement(product_id, 'STOCK_IN', quantity)
                else:
                    st.warning(
                        "Please select a product and enter a valid quantity!")

    # Record Sale
    elif choice == "Record Sale":
        st.subheader("Record Sale")
        inventory = view_inventory()
        if inventory.empty:
            st.warning("No products available to sell.")
        else:
            # Correctly create the product options dictionary
            product_options = {row["Product"]: row["ID"]
                               for _, row in inventory.iterrows()}

            # Select the product to sell
            product_name = st.selectbox(
                "Select Product", inventory["Product"].tolist())

            # Get the product ID from the selected product name
            product_id = product_options.get(product_name)

            # Get the quantity to sell
            quantity = st.number_input("Quantity sold", min_value=1)

            # If button is pressed, record the sale
            if st.button("Record Sale"):
                if product_name and quantity:
                    record_movement(product_id, 'SALE', quantity)
                else:
                    st.warning(
                        "Please select a product and enter a valid quantity!")

    # View Inventory
    elif choice == "View Inventory":
        st.subheader("Current Inventory")
        inventory = view_inventory()
        if not inventory.empty:
            st.dataframe(inventory)
        else:
            st.warning("No products found in inventory.")

    # View Movements
    elif choice == "View Movements":
        st.subheader("View Movement History")
        inventory = view_inventory()
        if inventory.empty:
            st.warning("No products available to view movements.")
        else:
            # Correctly referencing the 'product_name' and 'ID' columns from the inventory DataFrame
            product_options = {row["Product"]: row["ID"]
                               for _, row in inventory.iterrows()}
            product_name = st.selectbox(
                "Select Product", inventory["Product"].tolist())

            # Getting the product ID based on the selected product name
            product_id = product_options.get(product_name)

            # Fetching movement data for the selected product
            title, movements = view_movements(product_id)
            if not movements.empty:
                st.subheader(title)
                st.dataframe(movements)
            else:
                st.warning("No movements found for the selected product.")

    # Delete Product
    elif choice == "Delete Product":
        st.subheader("Delete Product")
        inventory = view_inventory()
        if inventory.empty:
            st.warning("No products available to delete.")
        else:
            # Correctly create the product options dictionary
            product_options = {row["Product"]: row["ID"]
                               for _, row in inventory.iterrows()}

            # Allow user to select a product
            product_name = st.selectbox(
                "Select Product to Delete", inventory["Product"].tolist())

            # If the button is pressed, delete the selected product
            if st.button("Delete Product"):
                # Get the corresponding product ID from the dictionary
                product_id = product_options.get(product_name)
                if product_id:
                    delete_product(product_id)
                else:
                    st.warning("Product not found!")


if __name__ == "__main__":
    main()
