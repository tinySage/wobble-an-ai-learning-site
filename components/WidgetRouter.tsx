"use client";
import { Widget } from "./widgets";
import { PART3, Widget3 } from "./widgets3";
export default function WidgetRouter({ name }: { name: string }) { return PART3.includes(name) ? <Widget3 name={name} /> : <Widget name={name} />; }
