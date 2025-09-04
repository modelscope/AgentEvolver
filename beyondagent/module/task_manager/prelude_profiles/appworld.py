from beyondagent.module.task_manager.user_profiles import EnvEntity, EnvEntityOpt, TaskPreference, UserProfile


venmo = EnvEntity(
    name="Venmo",
    description="A mobile payment service to send, request, and manage money transactions.",
    attrs={
        "Account": "User's Venmo account details.",
        "Friends": "List of Venmo friends.",
        "Transactions": "History of sent and received payments."
    },
    opts=[
        EnvEntityOpt("accept_request", "Accept pending payment requests."),
        EnvEntityOpt("reject_request", "Reject payment requests."),
        EnvEntityOpt("send_money", "Send money to a user."),
        EnvEntityOpt("request_money", "Request money from a user."),
        EnvEntityOpt("like_transaction", "Like a Venmo transaction."),
        EnvEntityOpt("remind_request", "Send reminders for pending requests."),
        EnvEntityOpt("manage_friends", "Befriend or unfriend users on Venmo."),
        EnvEntityOpt("change_password", "Change account password.")
    ]
)

amazon = EnvEntity(
    name="Amazon",
    description="An online e-commerce platform for purchasing, returning, and reviewing products.",
    attrs={
        "Account": "User's Amazon account details.",
        "Cart": "Items currently in the shopping cart.",
        "Wishlist": "Products saved for later purchase.",
        "Orders": "Purchase history and delivery details."
    },
    opts=[
        EnvEntityOpt("place_order", "Place an order for products."),
        EnvEntityOpt("return_item", "Initiate a return process."),
        EnvEntityOpt("manage_cart", "Add, remove, or move items in the cart."),
        EnvEntityOpt("manage_wishlist", "Add or remove items from wishlist."),
        EnvEntityOpt("post_question", "Post a product-related question."),
        EnvEntityOpt("write_review", "Write or update a product review."),
        EnvEntityOpt("check_delivery", "Check delivery date for an order.")
    ]
)

spotify = EnvEntity(
    name="Spotify",
    description="A music streaming service with song, album, and playlist management.",
    attrs={
        "Song Library": "Songs saved by the user.",
        "Album Library": "Albums saved by the user.",
        "Playlists": "User-created or followed playlists."
    },
    opts=[
        EnvEntityOpt("play_song", "Play a song, album, or playlist."),
        EnvEntityOpt("like_song", "Like songs or albums."),
        EnvEntityOpt("unfollow_artist", "Unfollow an artist."),
        EnvEntityOpt("follow_artist", "Follow an artist."),
        EnvEntityOpt("create_playlist", "Create a new playlist."),
        EnvEntityOpt("remove_song", "Remove songs from library or playlist."),
        EnvEntityOpt("export_library", "Export song/album/playlist data.")
    ]
)

gmail = EnvEntity(
    name="Gmail",
    description="An email service for sending, receiving, labeling, and managing emails.",
    attrs={
        "Inbox": "List of received email threads.",
        "Outbox": "List of sent email threads.",
        "Labels": "Custom labels to organize emails."
    },
    opts=[
        EnvEntityOpt("send_email", "Send an email."),
        EnvEntityOpt("forward_email", "Forward an email."),
        EnvEntityOpt("reply_email", "Reply to an email."),
        EnvEntityOpt("delete_email", "Delete emails."),
        EnvEntityOpt("label_email", "Label emails."),
        EnvEntityOpt("star_email", "Star or unstar email threads.")
    ]
)

simplenote = EnvEntity(
    name="Simple Note",
    description="A note-taking app for storing and managing notes and lists.",
    attrs={
        "Notes": "Collection of user notes.",
        "Tags": "Tags for organizing notes."
    },
    opts=[
        EnvEntityOpt("export_note", "Export notes."),
        EnvEntityOpt("update_note", "Update or edit a note."),
        EnvEntityOpt("add_note", "Add a new note."),
        EnvEntityOpt("note_to_playlist", "Create a playlist from note content.")
    ]
)

phone = EnvEntity(
    name="Phone",
    description="A mobile device for calls, text messages, voice messages, and alarms.",
    attrs={
        "Contacts": "List of saved contacts.",
        "Messages": "Text and voice messages.",
        "Alarms": "Configured alarms on the device."
    },
    opts=[
        EnvEntityOpt("send_text", "Send a text message."),
        EnvEntityOpt("send_voice", "Send a voice message."),
        EnvEntityOpt("set_alarm", "Set or update an alarm.")
    ]
)

todoist = EnvEntity(
    name="Todoist",
    description="A task management and to-do list application.",
    attrs={
        "Projects": "User's task projects.",
        "Tasks": "Individual tasks in projects."
    },
    opts=[
        EnvEntityOpt("complete_task", "Complete tasks."),
        EnvEntityOpt("update_task", "Update tasks."),
        EnvEntityOpt("move_task", "Move tasks between projects.")
    ]
)

splitwise = EnvEntity(
    name="Splitwise",
    description="An app for tracking shared expenses and balances.",
    attrs={
        "Groups": "Expense sharing groups.",
        "Expenses": "List of shared expenses."
    },
    opts=[
        EnvEntityOpt("add_expense", "Add a shared expense."),
        EnvEntityOpt("settle_expense", "Settle or record payments.")
    ]
)

filesystem = EnvEntity(
    name="File System",
    description="A local file storage system for managing files and directories.",
    attrs={
        "Directories": "Folder structure for files.",
        "Files": "Stored documents, images, and other file types."
    },
    opts=[
        EnvEntityOpt("download_file", "Download a file."),
        EnvEntityOpt("move_file", "Move files."),
        EnvEntityOpt("compress_file", "Compress files."),
        EnvEntityOpt("delete_file", "Delete files."),
        EnvEntityOpt("reorganize_files", "Change file organization structure.")
    ]
)




user_profile = UserProfile(
    name="Bob",
    background="A general computer user.",
    task=TaskPreference(
        num_entities=2,
        num_opts=3,
        relation_difficulty=3,
    )
)
user_profile.reg_entities([venmo, amazon, spotify, gmail, simplenote, phone, todoist, splitwise, filesystem])
