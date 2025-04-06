import sqlite3
import streamlit as st
from datetime import datetime

# Initialize database with two tables


def init_db():
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

# Add new product


def add_product(name):
    conn = sqlite3.connect("inventory.sqlite")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO products (name) VALUES (?)", (name,))
        conn.commit()
        st.success(f"Added product: {name}")
    except sqlite3.IntegrityError:
        st.warning(f"Product '{name}' already exists!")
    finally:
        conn.close()

# Record movement and update stock


def record_movement(product_id, movement_type, quantity):
    conn = sqlite3.connect("inventory.sqlite")
    cursor = conn.cursor()

    try:
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
    finally:
        conn.close()

# View current inventory


def view_inventory():
    conn = sqlite3.connect("inventory.sqlite")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, name, current_stock 
    FROM products
    ORDER BY name
    """)

    products = cursor.fetchall()
    conn.close()
    return products

# View movement history


def view_movements(product_id=None):
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

    movements = cursor.fetchall()
    conn.close()
    return title, movements

# Delete product


def delete_product(product_id):
    conn = sqlite3.connect("inventory.sqlite")
    cursor = conn.cursor()
    try:
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
        product_options = {product[1]: product[0] for product in inventory}
        product_name = st.selectbox(
            "Select Product", list(product_options.keys()))
        quantity = st.number_input("Quantity to add", min_value=1)
        if st.button("Record Stock In"):
            if product_name:
                product_id = product_options[product_name]
                record_movement(product_id, 'STOCK_IN', quantity)
            else:
                st.warning("Please select a product!")

    # Record Sale
    elif choice == "Record Sale":
        st.subheader("Record Sale")
        inventory = view_inventory()
        product_options = {product[1]: product[0] for product in inventory}
        product_name = st.selectbox(
            "Select Product", list(product_options.keys()))
        quantity = st.number_input("Quantity sold", min_value=1)
        if st.button("Record Sale"):
            if product_name:
                product_id = product_options[product_name]
                record_movement(product_id, 'SALE', quantity)
            else:
                st.warning("Please select a product!")

    # View Inventory
    elif choice == "View Inventory":
        st.subheader("Current Inventory")
        inventory = view_inventory()
        st.write(f"{'ID':<5}{'Product':<20}{'Stock':<10}")
        st.write("-" * 35)
        for row in inventory:
            st.write(f"{row[0]:<5}{row[1]:<20}{row[2]:<10}")

    # View Movements
    elif choice == "View Movements":
        st.subheader("View Movement History")
        inventory = view_inventory()
        product_options = {product[1]: product[0] for product in inventory}
        product_name = st.selectbox(
            "Select Product", list(product_options.keys()))
        product_id = product_options[product_name]
        title, movements = view_movements(product_id)
        st.write(title)
        st.write(f"{'Type':<10}{'Qty':<5}{'Product':<15}{'Time':<20}")
        st.write("-" * 50)
        for row in movements:
            st.write(f"{row[0]:<10}{row[1]:<5}{row[3]:<15}{row[2]:<20}")

    # Delete Product
    elif choice == "Delete Product":
        st.subheader("Delete Product")
        inventory = view_inventory()
        product_options = {product[1]: product[0] for product in inventory}
        product_name = st.selectbox(
            "Select Product to Delete", list(product_options.keys()))
        if st.button("Delete Product"):
            product_id = product_options[product_name]
            delete_product(product_id)


if __name__ == "__main__":
    main()
