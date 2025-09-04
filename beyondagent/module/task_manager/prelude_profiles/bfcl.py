from dataclasses import dataclass
from typing import List
from datetime import date

from beyondagent.module.task_manager.user_profiles import EnvEntity, EnvEntityOpt, TaskPreference, UserProfile

# 假设 CRUD 操作
def get_standard_file_ops():
    return [
        EnvEntityOpt("create", "Create a new file or directory."),
        EnvEntityOpt("read", "Read contents of a file or list directory."),
        EnvEntityOpt("update", "Modify contents or metadata of a file."),
        EnvEntityOpt("delete", "Delete a file or directory."),
        EnvEntityOpt("move_copy", "Move or copy files and directories."),
        EnvEntityOpt("search", "Search for files or content."),
        EnvEntityOpt("compare", "Compare two files and show differences."),
    ]

def get_vehicle_ops():
    return [
        EnvEntityOpt("start_engine", "Start the vehicle engine."),
        EnvEntityOpt("stop_engine", "Stop the vehicle engine."),
        EnvEntityOpt("refuel", "Add fuel to the vehicle."),
        EnvEntityOpt("check_tire_pressure", "Check the tire pressure."),
        EnvEntityOpt("lock_unlock_doors", "Lock or unlock vehicle doors."),
        EnvEntityOpt("set_navigation", "Set the vehicle navigation system."),
        EnvEntityOpt("check_battery", "Check the vehicle battery status.")
    ]

def get_flight_ops():
    return [
        EnvEntityOpt("check_flight_cost", "Retrieve flight cost for given route."),
        EnvEntityOpt("book_flight", "Book a flight ticket."),
        EnvEntityOpt("cancel_flight", "Cancel an existing flight booking."),
        EnvEntityOpt("purchase_insurance", "Buy travel insurance for a booking.")
    ]

def get_social_media_ops():
    return [
        EnvEntityOpt("post", "Post a new message."),
        EnvEntityOpt("retweet", "Retweet an existing post."),
        EnvEntityOpt("comment", "Comment on a post."),
        EnvEntityOpt("delete_message", "Delete a sent message."),
        EnvEntityOpt("view_sent_messages", "List all sent messages.")
    ]

def get_trading_ops():
    return [
        EnvEntityOpt("get_stock_info", "Retrieve details of a stock."),
        EnvEntityOpt("add_watchlist", "Add a stock to watchlist."),
        EnvEntityOpt("remove_watchlist", "Remove a stock from watchlist."),
        EnvEntityOpt("place_order", "Place a buy or sell order."),
        EnvEntityOpt("cancel_order", "Cancel an existing order."),
        EnvEntityOpt("get_account_details", "Retrieve account balance and linked cards.")
    ]

def get_support_ops():
    return [
        EnvEntityOpt("create_ticket", "Create a new support ticket."),
        EnvEntityOpt("update_ticket", "Update ticket details."),
        EnvEntityOpt("close_ticket", "Close a support ticket."),
        EnvEntityOpt("view_ticket", "View details of a support ticket.")
    ]

# 创建实体
entities = [
    EnvEntity(
        name="FileSystem",
        description="Represents the user's file system for file and directory management.",
        attrs={
            "directory_structure": "Hierarchy of folders and files.",
            "file_metadata": "File names, sizes, modification times.",
            "permissions": "Access rights for files and directories."
        },
        opts=get_standard_file_ops()
    ),
    EnvEntity(
        name="Vehicle",
        description="Represents a vehicle with controllable systems for travel readiness.",
        attrs={
            "fuel_level": "Current fuel in the tank.",
            "tire_pressure": "Pressure of each tire.",
            "door_lock_status": "Whether doors are locked or unlocked.",
            "battery_voltage": "Current battery voltage.",
            "ac_settings": "Air conditioning temperature and fan speed."
        },
        opts=get_vehicle_ops()
    ),
    EnvEntity(
        name="FlightBookingSystem",
        description="System for searching, booking, and managing flights.",
        attrs={
            "departure_airport": "Code of the departure airport.",
            "arrival_airport": "Code of the arrival airport.",
            "class_type": "Travel class (economy, business, first).",
            "flight_cost": "Cost of the flight ticket.",
            "booking_id": "Unique booking identifier."
        },
        opts=get_flight_ops()
    ),
    EnvEntity(
        name="SocialMediaAccount",
        description="Platform for posting updates, retweeting, and interacting with others.",
        attrs={
            "username": "Account username.",
            "followers_count": "Number of followers.",
            "posts": "List of published posts."
        },
        opts=get_social_media_ops()
    ),
    EnvEntity(
        name="TradingAccount",
        description="Represents a stock trading account for investments.",
        attrs={
            "balance": "Available account balance.",
            "watchlist": "List of stocks being monitored.",
            "order_history": "Record of past and current orders."
        },
        opts=get_trading_ops()
    ),
    EnvEntity(
        name="CustomerSupportTicket",
        description="System for tracking and resolving user-reported issues.",
        attrs={
            "ticket_id": "Unique identifier for the ticket.",
            "title": "Ticket title.",
            "description": "Detailed description of the issue.",
            "priority": "Ticket priority level.",
            "status": "Current status of the ticket."
        },
        opts=get_support_ops()
    )
]

# 定义用户和任务偏好
task_pref = TaskPreference(num_entities=2, num_opts=3, relation_difficulty=3)

# 创建用户配置文件
user_profile = UserProfile(
    name="Alice",
    background="A general user.",
    task=task_pref
)

# 注册实体
user_profile.reg_entities(entities)

