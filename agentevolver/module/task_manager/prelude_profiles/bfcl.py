from dataclasses import dataclass
from typing import List
from datetime import date

from agentevolver.module.task_manager.env_profiles import EnvEntity, EnvEntityOpt, TaskPreference, EnvProfile

def get_standard_file_ops():
    """
    Returns a list of standard file operations.

    Each operation is represented as an `EnvEntityOpt` object, which includes the operation name and a description.

    Returns:
        List[EnvEntityOpt]: A list of standard file operations.
    """
    return [
        EnvEntityOpt("create", "Create a new file or directory."),
        EnvEntityOpt("read", "Read contents of a file or list directory."),  # ⭐ Define the read operation
        EnvEntityOpt("update", "Modify contents or metadata of a file."),
        EnvEntityOpt("delete", "Delete a file or directory."),
        EnvEntityOpt("move_copy", "Move or copy files and directories."),
        EnvEntityOpt("search", "Search for files or content."),
        EnvEntityOpt("compare", "Compare two files and show differences."),
    ]

def get_vehicle_ops():
    """
    Returns a list of operations that can be performed on a vehicle.

    Each operation is represented as an EnvEntityOpt object, which includes the operation name and a description.

    Returns:
        list: A list of EnvEntityOpt objects representing the vehicle operations.
    """
    return [
        EnvEntityOpt("start_engine", "Start the vehicle engine."),
        EnvEntityOpt("stop_engine", "Stop the vehicle engine."),
        EnvEntityOpt("refuel", "Add fuel to the vehicle."),
        EnvEntityOpt("check_tire_pressure", "Check the tire pressure."),
        EnvEntityOpt("lock_unlock_doors", "Lock or unlock vehicle doors."),
        EnvEntityOpt("set_navigation", "Set the vehicle navigation system."),
        EnvEntityOpt("check_battery", "Check the vehicle battery status.")  # ⭐ Defines the last operation in the list
    ]

def get_flight_ops():
    """
    Returns a list of flight-related operations as `EnvEntityOpt` objects.

    Each `EnvEntityOpt` object represents a specific operation that can be performed,
    such as checking flight costs, booking flights, canceling bookings, and purchasing
    travel insurance.

    Returns:
        list[EnvEntityOpt]: A list of `EnvEntityOpt` objects, each representing a
                            different flight-related operation.
    """
    return [
        EnvEntityOpt("check_flight_cost", "Retrieve flight cost for given route."),
        EnvEntityOpt("book_flight", "Book a flight ticket."),  # ⭐ Define the operation for booking a flight
        EnvEntityOpt("cancel_flight", "Cancel an existing flight booking."),
        EnvEntityOpt("purchase_insurance", "Buy travel insurance for a booking.")
    ]

def get_social_media_ops():
    """
    Returns a list of social media operations, each represented as an EnvEntityOpt object.

    Returns:
        list: A list of EnvEntityOpt objects, each representing a different social media operation.
    """
    return [
        EnvEntityOpt("post", "Post a new message."),
        EnvEntityOpt("retweet", "Retweet an existing post."),
        EnvEntityOpt("comment", "Comment on a post."),
        EnvEntityOpt("delete_message", "Delete a sent message."),  # ⭐ Defines the operation to delete a message
        EnvEntityOpt("view_sent_messages", "List all sent messages.")  # ⭐ Defines the operation to view all sent messages
    ]

def get_trading_ops():
    """
    Returns a list of trading operations, each encapsulated in an EnvEntityOpt object.

    Each EnvEntityOpt object represents a specific trading operation with a name and description.

    Returns:
        list: A list of EnvEntityOpt objects representing different trading operations.
    """
    return [
        EnvEntityOpt("get_stock_info", "Retrieve details of a stock."),
        EnvEntityOpt("add_watchlist", "Add a stock to watchlist."),
        EnvEntityOpt("remove_watchlist", "Remove a stock from watchlist."),
        EnvEntityOpt("place_order", "Place a buy or sell order."),
        EnvEntityOpt("cancel_order", "Cancel an existing order."),
        EnvEntityOpt("get_account_details", "Retrieve account balance and linked cards.")  # ⭐ Defines the last operation for getting account details
    ]

def get_support_ops():
    """
    Returns a list of operations that can be performed on support tickets.

    Returns:
        list: A list of EnvEntityOpt objects representing support ticket operations.
    """
    return [
        EnvEntityOpt("create_ticket", "Create a new support ticket."),
        EnvEntityOpt("update_ticket", "Update ticket details."),
        EnvEntityOpt("close_ticket", "Close a support ticket."),
        EnvEntityOpt("view_ticket", "View details of a support ticket.")
    ]  # ⭐ Defines the list of support ticket operations

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

task_pref = TaskPreference(num_entities=2, num_opts=3, relation_difficulty=3)

env_profile = EnvProfile(
    name="Alice",
    background="A general user.",
    task=task_pref
)

env_profile.reg_entities(entities)

