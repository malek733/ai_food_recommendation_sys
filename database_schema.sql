-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.categories (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  restaurant_id uuid NOT NULL,
  name jsonb NOT NULL,
  description jsonb,
  is_active boolean NOT NULL DEFAULT true,
  sort_order integer DEFAULT 0,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT categories_pkey PRIMARY KEY (id),
  CONSTRAINT categories_restaurant_id_fkey FOREIGN KEY (restaurant_id) REFERENCES public.restaurants(id)
);
CREATE TABLE public.locations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  name jsonb NOT NULL,
  description jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  is_default boolean NOT NULL DEFAULT false,
  CONSTRAINT locations_pkey PRIMARY KEY (id)
);
CREATE TABLE public.menu_item_options (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  menu_item_id uuid NOT NULL,
  name jsonb NOT NULL,
  type USER-DEFINED NOT NULL,
  is_required boolean NOT NULL DEFAULT false,
  sort_order integer DEFAULT 0,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT menu_item_options_pkey PRIMARY KEY (id),
  CONSTRAINT menu_item_options_menu_item_id_fkey FOREIGN KEY (menu_item_id) REFERENCES public.menu_items(id)
);
CREATE TABLE public.menu_items (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  category_id uuid NOT NULL,
  name jsonb NOT NULL,
  description jsonb,
  price numeric NOT NULL,
  image_url text,
  is_available boolean NOT NULL DEFAULT true,
  sort_order integer DEFAULT 0,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  preparation_time integer,
  available_quantity integer CHECK (available_quantity IS NULL OR available_quantity >= 0),
  CONSTRAINT menu_items_pkey PRIMARY KEY (id),
  CONSTRAINT menu_items_category_id_fkey FOREIGN KEY (category_id) REFERENCES public.categories(id)
);
CREATE TABLE public.option_choices (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  option_id uuid NOT NULL,
  name jsonb NOT NULL,
  price_modifier numeric DEFAULT 0,
  sort_order integer DEFAULT 0,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT option_choices_pkey PRIMARY KEY (id),
  CONSTRAINT option_choices_option_id_fkey FOREIGN KEY (option_id) REFERENCES public.menu_item_options(id)
);
CREATE TABLE public.order_items (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  order_id uuid NOT NULL,
  menu_item_id uuid NOT NULL,
  quantity integer NOT NULL DEFAULT 1,
  unit_price numeric NOT NULL,
  total_price numeric NOT NULL,
  selected_options jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT order_items_pkey PRIMARY KEY (id),
  CONSTRAINT order_items_menu_item_id_fkey FOREIGN KEY (menu_item_id) REFERENCES public.menu_items(id),
  CONSTRAINT order_items_order_id_fkey FOREIGN KEY (order_id) REFERENCES public.orders(id)
);
CREATE TABLE public.orders (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid,
  total_amount numeric NOT NULL,
  status USER-DEFINED NOT NULL DEFAULT 'open'::order_status,
  customer_name text NOT NULL,
  customer_phone text NOT NULL,
  delivery_address text,
  notes text,
  payment_intent_id text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  customer_email text,
  payment_status text DEFAULT 'pending'::text,
  payment_method text DEFAULT 'cash_on_delivery'::text,
  sadad_transaction_id text,
  CONSTRAINT orders_pkey PRIMARY KEY (id),
  CONSTRAINT orders_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.payments (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  order_id uuid NOT NULL,
  sadad_transaction_id text,
  amount numeric NOT NULL,
  status text NOT NULL DEFAULT 'pending'::text,
  payment_method text NOT NULL DEFAULT 'sadad'::text,
  sadad_merchant_id text,
  sadad_response_data jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  failure_reason text,
  transaction_timestamp timestamp with time zone,
  callback_received_at timestamp with time zone,
  CONSTRAINT payments_pkey PRIMARY KEY (id),
  CONSTRAINT payments_order_id_fkey FOREIGN KEY (order_id) REFERENCES public.orders(id)
);
CREATE TABLE public.profiles (
  id uuid NOT NULL,
  email text NOT NULL,
  full_name text,
  phone text,
  role USER-DEFINED NOT NULL DEFAULT 'customer'::user_role,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT profiles_pkey PRIMARY KEY (id),
  CONSTRAINT profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);
CREATE TABLE public.restaurants (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  name jsonb NOT NULL,
  description jsonb,
  logo_url text,
  cover_image_url text,
  is_active boolean NOT NULL DEFAULT true,
  sort_order integer DEFAULT 0,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  category ARRAY DEFAULT ARRAY['general'::text],
  coming_soon boolean NOT NULL DEFAULT false,
  location_id uuid NOT NULL,
  CONSTRAINT restaurants_pkey PRIMARY KEY (id),
  CONSTRAINT restaurants_location_id_fkey FOREIGN KEY (location_id) REFERENCES public.locations(id)
);
CREATE TABLE public.site_settings (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  logo_en_url text,
  logo_ar_url text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  hero_image_url text,
  footer_background_url text,
  about_hero_url text,
  default_placeholder_url text,
  minimum_delivery_amount numeric NOT NULL DEFAULT 0,
  CONSTRAINT site_settings_pkey PRIMARY KEY (id)
);
CREATE TABLE public.user_invitations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  email text NOT NULL,
  full_name text,
  phone text,
  role USER-DEFINED NOT NULL DEFAULT 'restaurant_manager'::user_role,
  restaurant_assignments jsonb NOT NULL DEFAULT '[]'::jsonb,
  invited_by uuid NOT NULL,
  invitation_token text NOT NULL UNIQUE,
  status text NOT NULL DEFAULT 'pending'::text CHECK (status = ANY (ARRAY['pending'::text, 'accepted'::text, 'expired'::text, 'cancelled'::text])),
  expires_at timestamp with time zone NOT NULL DEFAULT (now() + '7 days'::interval),
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT user_invitations_pkey PRIMARY KEY (id),
  CONSTRAINT user_invitations_invited_by_fkey FOREIGN KEY (invited_by) REFERENCES public.profiles(id)
);
CREATE TABLE public.user_restaurant_assignments (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  restaurant_id uuid NOT NULL,
  assigned_by uuid NOT NULL,
  assigned_at timestamp with time zone NOT NULL DEFAULT now(),
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT user_restaurant_assignments_pkey PRIMARY KEY (id),
  CONSTRAINT user_restaurant_assignments_assigned_by_fkey FOREIGN KEY (assigned_by) REFERENCES public.profiles(id),
  CONSTRAINT user_restaurant_assignments_restaurant_id_fkey FOREIGN KEY (restaurant_id) REFERENCES public.restaurants(id),
  CONSTRAINT user_restaurant_assignments_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.profiles(id)
);
CREATE TABLE public.users (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  email text NOT NULL UNIQUE,
  first_name text NOT NULL,
  last_name text NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT timezone('utc'::text, now()),
  CONSTRAINT users_pkey PRIMARY KEY (id)
);